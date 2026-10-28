#!/usr/bin/env python3
"""Run PAHC-3DGS compression."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.backend import load_trained_scene
from src.codec import (
    CompactScene,
    TemplateInstance,
    decode_compact_scene,
    expand_compact_basis,
    load_compact_scene,
    save_compact_scene,
)
from src.compression import (
    ClusterConfig,
    MatchingConfig,
    PAHCConfig,
    RefinementConfig,
    run_geo32_compression,
)
from src.quantization import quantize_appearance_attributes


def load_config(path):
    cfg_raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = PAHCConfig(
        feature_dim=int(cfg_raw.get("feature_dim", 32)),
        feature_iteration=int(cfg_raw.get("feature_iteration", 10000)),
        clustering=ClusterConfig(**cfg_raw.get("clustering", {})),
        matching=MatchingConfig(**cfg_raw.get("matching", {})),
        refinement=RefinementConfig(**cfg_raw.get("refinement", {})),
    )
    return cfg_raw, config


def codebook_payload(codebooks):
    return {
        name: {
            "centers": codebook.centers.cpu(),
            "indices": codebook.indices.cpu(),
            "value_shape": codebook.value_shape,
        }
        for name, codebook in codebooks.items()
    }


def quantized_attribute_names(appearance_cfg):
    aliases = {
        "dc": "features_dc",
        "sh": "features_rest",
        "scale": "scaling",
        "rot": "rotation",
        "rotation": "rotation",
        "opacity": "opacity",
    }
    configured = appearance_cfg.get(
        "quantized_attributes",
        ["dc", "sh", "scale", "rotation", "opacity"],
    )
    return [aliases[name] for name in configured]


def build_real_compact_scene(gaussians, result, appearance_cfg):
    original_count = int(gaussians._xyz.shape[0])
    remove_mask = result["pts_to_remove_mask"]
    if remove_mask is None:
        remove_mask = torch.zeros(original_count, dtype=torch.bool, device=gaussians._xyz.device)
    keep_mask = ~remove_mask
    kept_indices = torch.where(keep_mask)[0]
    old_to_base = torch.full((original_count,), -1, dtype=torch.long, device=gaussians._xyz.device)
    old_to_base[kept_indices] = torch.arange(kept_indices.shape[0], device=gaussians._xyz.device)

    templates = {}
    instances = []
    template_ids = {}
    for alignment in result["alignments"]:
        src = int(alignment["src"])
        if src not in template_ids:
            template_id = len(template_ids)
            template_ids[src] = template_id
            source_indices = torch.where(alignment["src_mask"])[0]
            templates[str(template_id)] = {
                "indices": old_to_base[source_indices].cpu(),
                "source_label": src,
            }
        transform = alignment["transform_net"]
        instances.append(
            TemplateInstance(
                template_id=template_ids[src],
                instance_id=int(alignment["tgt"]),
                rotation=torch.nn.functional.normalize(transform.rotation_q.detach(), dim=0).cpu(),
                translation=transform.translation.detach().cpu(),
                scale=torch.abs(transform.scale_factor.detach()).cpu(),
                center=transform.src_center.detach().cpu(),
            )
        )

    gaussians.remove_gaussians(remove_mask)
    basis_tensors = current_gaussian_tensors(gaussians)
    attributes = {
        "features_dc": gaussians._features_dc.detach(),
        "features_rest": gaussians._features_rest.detach(),
        "scaling": gaussians._scaling.detach(),
        "rotation": gaussians._rotation.detach(),
        "opacity": gaussians._opacity.detach(),
    }
    quantized_names = set(quantized_attribute_names(appearance_cfg))
    quantized_attributes = {
        name: values for name, values in attributes.items() if name in quantized_names
    }
    configured_sizes = appearance_cfg.get("codebook_size", {})
    codebook_sizes = {
        "features_dc": configured_sizes.get("dc", 4096),
        "features_rest": configured_sizes.get("sh", 4096),
        "scaling": configured_sizes.get("scale", 4096),
        "rotation": configured_sizes.get("rotation", 4096),
        "opacity": configured_sizes.get("opacity", 4096),
    }
    qat_enabled = bool(appearance_cfg.get("qat", {}).get("enabled", False))
    codebooks = {}
    if not qat_enabled:
        with torch.no_grad():
            codebooks = quantize_appearance_attributes(
                quantized_attributes,
                codebook_sizes=codebook_sizes,
                num_iters=int(appearance_cfg.get("num_iters", 10)),
            )
    static_gaussians = {"xyz": gaussians._xyz.detach().cpu()}
    for name, values in attributes.items():
        if qat_enabled or name not in quantized_names:
            static_gaussians[name] = values.cpu()
    compact = CompactScene(
        templates=templates,
        static_gaussians=static_gaussians,
        instances=instances,
        codebooks=codebook_payload(codebooks),
        metadata={
            "original_count": original_count,
            "basis_count": int(gaussians._xyz.shape[0]),
            "removed_count": int(remove_mask.sum().item()),
            "num_clusters": len(result["cluster_props"]),
            "num_matches": len(result["matches"]),
            "num_alignments": len(result["alignments"]),
            "num_templates": len(templates),
            "num_instances": len(instances),
            "quantized_attributes": sorted(quantized_names),
            "parameter_space": "raw",
            "quantized_evaluation": "saved_package_independent_decode",
        },
    )
    return compact, basis_tensors


def make_gaussian_model(backend, tensors, sh_degree, active_sh_degree, device):
    model = backend.GaussianModel(sh_degree)
    model._xyz = nn.Parameter(tensors["xyz"].to(device))
    model._features_dc = nn.Parameter(tensors["features_dc"].to(device))
    model._features_rest = nn.Parameter(tensors["features_rest"].to(device))
    model._scaling = nn.Parameter(tensors["scaling"].to(device))
    model._rotation = nn.Parameter(tensors["rotation"].to(device))
    model._opacity = nn.Parameter(tensors["opacity"].to(device))
    model._mask = torch.ones((model._xyz.shape[0], 1), device=device)
    model.max_radii2D = torch.zeros(model._xyz.shape[0], device=device)
    model.xyz_gradient_accum = torch.zeros((model._xyz.shape[0], 1), device=device)
    model.denom = torch.zeros((model._xyz.shape[0], 1), device=device)
    model.active_sh_degree = active_sh_degree
    return model


def current_gaussian_tensors(gaussians):
    return {
        "xyz": gaussians._xyz.detach().clone(),
        "features_dc": gaussians._features_dc.detach().clone(),
        "features_rest": gaussians._features_rest.detach().clone(),
        "scaling": gaussians._scaling.detach().clone(),
        "rotation": gaussians._rotation.detach().clone(),
        "opacity": gaussians._opacity.detach().clone(),
    }


def append_payloads(tensors, payloads):
    if not payloads:
        return tensors
    merged = {}
    for name, values in tensors.items():
        merged[name] = torch.cat([values] + [payload[name].detach() for payload in payloads], dim=0)
    return merged


def evaluate_rendering(scene, gaussians, pipe, backend, white_background):
    device = gaussians._xyz.device
    background = torch.tensor(
        [1, 1, 1] if white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    render_times = []
    cameras = scene.getTrainCameras()
    for camera in tqdm(cameras, desc="  Rendering views", unit="view"):

        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            image = backend.render(camera, gaussians, pipe, background)["render"].clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            render_times.append(time.time() - start)
            target = camera.original_image.to(device).clamp(0, 1)
            psnr_sum += float(backend.psnr(image, target).mean().item())
            ssim_sum += float(backend.ssim(image, target).mean().item())
            lpips_sum += float(backend.lpips(image, target, net_type="vgg").mean().item())
    count = len(cameras)
    average_seconds = sum(render_times) / max(count, 1)
    return {
        "views": count,
        "psnr": psnr_sum / max(count, 1),
        "ssim": ssim_sum / max(count, 1),
        "lpips": lpips_sum / max(count, 1),
        "render_ms": average_seconds * 1000,
        "fps": 1.0 / max(average_seconds, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pahc.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/scene.pahc.pt"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--saga-root", type=Path, default=None)
    parser.add_argument("--feature-iteration", type=int, default=None)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--recluster", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    cfg_raw, config = load_config(args.config)
    appearance_cfg = cfg_raw.get("appearance", {})
    if args.model_path is None:
        parser.error("--model-path is required")
    feature_iteration = args.feature_iteration or config.feature_iteration
    print("=== Stage 1/4: Loading trained scene ===")
    loaded = load_trained_scene(
        args.model_path,
        source_path=args.source_path,
        feature_iteration=feature_iteration,
        feature_dim=config.feature_dim,
        saga_root=args.saga_root,
    )
    print(f"  Loaded {loaded['gaussians']._xyz.shape[0]} Gaussians, "
          f"{len(loaded['scene'].getTrainCameras())} train cameras")

    legacy_labels_path = args.model_path / "micro_cluster_labels.pt"
    if args.labels_path:
        labels_path = args.labels_path
    elif legacy_labels_path.exists() and not args.recluster:
        labels_path = legacy_labels_path
    else:
        labels_path = args.output.with_suffix(".labels.pt")
    labels = None
    if labels_path.exists() and not args.recluster:
        print(f"  Loading cached labels from {labels_path}")
        labels = torch.load(labels_path, map_location=loaded["features"].device)
    print("=== Stage 2/4: Geometry compression ===")
    result = run_geo32_compression(
        loaded["scene"],
        loaded["gaussians"],
        loaded["features"],
        loaded["pipe"],
        loaded["backend"],
        config=config,
        labels=labels,
    )
    if labels is None:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result["labels"].cpu(), labels_path)
        print(f"  Saved labels to {labels_path}")
    print("=== Stage 3/4: Building compact scene + appearance quantization ===")
    compact, basis_tensors = build_real_compact_scene(
        loaded["gaussians"],
        result,
        appearance_cfg,
    )
    qat_cfg = dict(appearance_cfg.get("qat", {}))
    if qat_cfg.get("enabled", False):
        from src.qat import fit_qat
        sizes = appearance_cfg.get("codebook_size", {})
        qat_cfg["codebook_sizes"] = {
            "features_dc": sizes.get("dc", 4096),
            "features_rest": sizes.get("sh", 4096),
            "scaling": sizes.get("scale", 4096),
            "rotation": sizes.get("rotation", 4096),
        }
        print("  Refining shared basis with four-group STE QAT...")
        compact = fit_qat(compact, basis_tensors, loaded, qat_cfg,
                          args.output.with_suffix(".qat"))
    compact.metadata["quantized_evaluation"] = "saved_package_independent_decode"
    compact.metadata["model_path"] = str(args.model_path.resolve())
    compact.metadata["feature_iteration"] = feature_iteration
    lowbit = bool(cfg_raw.get("codec", {}).get("lowbit", False))
    save_compact_scene(compact, args.output, lowbit=lowbit)
    stored_scene = load_compact_scene(args.output)
    if not args.skip_evaluation:
        print("=== Stage 4/4: Rendering evaluation ===")
        device = loaded["gaussians"]._xyz.device
        unquantized_tensors = expand_compact_basis(compact, basis_tensors)
        unquantized_model = make_gaussian_model(
            loaded["backend"],
            unquantized_tensors,
            loaded["dataset"].sh_degree,
            loaded["gaussians"].active_sh_degree,
            device,
        )
        print("  Evaluating unquantized model...")
        compact.metadata["unquantized_metrics"] = evaluate_rendering(
            loaded["scene"],
            unquantized_model,
            loaded["pipe"],
            loaded["backend"],
            loaded["dataset"].white_background,
        )
        quantized_tensors = decode_compact_scene(stored_scene)
        quantized_model = make_gaussian_model(
            loaded["backend"],
            quantized_tensors,
            loaded["dataset"].sh_degree,
            loaded["gaussians"].active_sh_degree,
            device,
        )
        print("  Evaluating decoded stored model...")
        compact.metadata["quantized_metrics"] = evaluate_rendering(
            loaded["scene"],
            quantized_model,
            loaded["pipe"],
            loaded["backend"],
            loaded["dataset"].white_background,
        )

    # Serialize the same pre-storage tensors again, adding evaluation metadata.
    if not args.skip_evaluation:
        save_compact_scene(compact, args.output, lowbit=lowbit)
    print(json.dumps(compact.metadata, indent=2))
    print(f"Saved compact scene to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
