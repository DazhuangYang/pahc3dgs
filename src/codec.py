"""Compact PAHC-3DGS representation and storage helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .math_utils import quat_multiply, quat_to_rotmat, rotmat_to_quat


FORMAT_VERSION = "0.1"


@dataclass
class TemplateInstance:
    """Pose and template relation for one reconstructed instance."""

    template_id: int
    instance_id: int
    rotation: torch.Tensor
    translation: torch.Tensor
    scale: torch.Tensor | None = None
    center: torch.Tensor | None = None


@dataclass
class CompactScene:
    """Hierarchical compact representation used by PAHC-3DGS."""

    templates: dict[str, torch.Tensor]
    static_gaussians: dict[str, torch.Tensor] = field(default_factory=dict)
    instances: list[TemplateInstance] = field(default_factory=list)
    codebooks: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = FORMAT_VERSION

    def to_payload(self):
        return {
            "version": self.version,
            "templates": self.templates,
            "static_gaussians": self.static_gaussians,
            "instances": [
                {
                    "template_id": item.template_id,
                    "instance_id": item.instance_id,
                    "rotation": item.rotation,
                    "translation": item.translation,
                    "scale": item.scale,
                    "center": item.center,
                }
                for item in self.instances
            ],
            "codebooks": self.codebooks,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]):
        return cls(
            version=payload.get("version", "unknown"),
            templates=payload.get("templates", {}),
            static_gaussians=payload.get("static_gaussians", {}),
            instances=[
                TemplateInstance(
                    template_id=item["template_id"],
                    instance_id=item["instance_id"],
                    rotation=item["rotation"],
                    translation=item["translation"],
                    scale=item.get("scale"),
                    center=item.get("center"),
                )
                for item in payload.get("instances", [])
            ],
            codebooks=payload.get("codebooks", {}),
            metadata=payload.get("metadata", {}),
        )


def save_compact_scene(scene: CompactScene, path: str | Path, *, lowbit: bool = False):
    """Save a compact scene as a torch payload."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = scene.to_payload()
    if lowbit:
        from .lowbit import encode_payload
        payload = encode_payload(payload)
    torch.save(payload, path)


def load_compact_scene(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load a compact scene saved by :func:`save_compact_scene`."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "storage" in payload:
        from .lowbit import decode_payload
        payload = decode_payload(payload)
    return CompactScene.from_payload(payload)

def decode_compact_basis(scene: CompactScene):
    """Decode only the stored basis Gaussians."""

    base = {key: value.clone() for key, value in scene.static_gaussians.items()}
    for name, codebook in scene.codebooks.items():
        values = codebook["centers"][codebook["indices"].long()]
        value_shape = tuple(codebook.get("value_shape", values.shape[1:]))
        base[name] = values.reshape(values.shape[0], *value_shape)
    return base


def expand_compact_basis(scene: CompactScene, base):
    """Expand a basis into all template instances while preserving gradients."""

    chunks: list[dict[str, torch.Tensor]] = []
    if base:
        chunks.append(base)

    indexed_templates = bool(scene.templates) and all(
        isinstance(value, dict) and "indices" in value
        for value in scene.templates.values()
    )
    if indexed_templates:
        templates_by_id = {
            int(template_id): value["indices"].long()
            for template_id, value in scene.templates.items()
        }
    else:
        templates_by_id = {0: scene.templates}

    for instance in scene.instances:
        template = templates_by_id.get(instance.template_id)
        if isinstance(template, torch.Tensor):
            reference = next(iter(base.values()))
            template = template.to(device=reference.device)
            output = {key: value[template] for key, value in base.items()}
        else:
            output = {key: value.clone() for key, value in template.items()}
        if "xyz" in output:
            reference = output["xyz"]
            center = instance.center
            if center is None:
                center = output["xyz"].mean(dim=0)
            else:
                center = center.to(reference)
            rotation = instance.rotation.to(reference)
            if rotation.shape == (4,):
                rotation_matrix = quat_to_rotmat(rotation)
                global_quat = rotation
            else:
                rotation_matrix = rotation
                global_quat = rotmat_to_quat(rotation)
            translation = instance.translation.to(reference)
            if instance.scale is None:
                scale = reference.new_ones(1)
            else:
                scale = torch.abs(instance.scale.to(reference)).reshape(1)
            output["xyz"] = (output["xyz"] - center) * scale
            output["xyz"] = output["xyz"].matmul(rotation_matrix.t()) + center + translation
            if "rotation" in output:
                output["rotation"] = quat_multiply(global_quat.unsqueeze(0), output["rotation"])
            if "scaling" in output:
                if scene.metadata.get("parameter_space") == "raw":
                    # GaussianModel stores log-scales; legacy payloads keep their convention.
                    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                        raise ValueError("Instance scale must be finite and positive")
                    output["scaling"] = output["scaling"] + torch.log(scale)
                else:
                    output["scaling"] = output["scaling"] * scale
            if "features_rest" in output:
                output["features_rest"] = torch.zeros_like(output["features_rest"])
        chunks.append(output)
    if not chunks:
        return {}
    keys = chunks[0].keys()
    return {key: torch.cat([chunk[key] for chunk in chunks if key in chunk], dim=0) for key in keys}


def decode_compact_scene(scene: CompactScene):
    """Decode static templates and instances into Gaussian tensor dictionaries."""

    return expand_compact_basis(scene, decode_compact_basis(scene))

def estimate_payload_bytes(payload: Any):
    """Estimate memory size of nested tensor payloads."""

    if isinstance(payload, torch.Tensor):
        return int(payload.numel() * payload.element_size())
    if isinstance(payload, dict):
        return sum(estimate_payload_bytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(estimate_payload_bytes(value) for value in payload)
    return 0
