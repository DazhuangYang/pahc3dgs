"""Four-group STE QAT adapted from the verified Geo33/CompGS schedule.

This uses PAHC's existing KMeans implementation and instance expansion. It
preserves the original algorithm and schedule, not bitwise native CUDA results.
Low-bit serialization is a separate post-training operation.
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch

from .codec import expand_compact_basis
from .quantization import Codebook, fit_codebook, update_centers


GROUPS = ("features_rest", "features_dc", "rotation", "scaling")
FIELDS = {"xyz": "_xyz", "features_dc": "_features_dc",
          "features_rest": "_features_rest", "scaling": "_scaling",
          "rotation": "_rotation", "opacity": "_opacity"}


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(value)


def _quantized_forward(values, books, sizes, assign):
    """Refresh centers off-graph, then expose codewords with identity gradients."""
    result = dict(values)
    with torch.no_grad():
        for name in GROUPS:
            raw = values[name]
            if assign:
                previous = books.get(name)
                books[name] = fit_codebook(
                    raw, num_clusters=sizes[name], num_iters=1,
                    initial_centers=None if previous is None else previous.centers)
            else:
                book = books[name]
                centers = update_centers(raw.detach().reshape(len(raw), -1),
                                         book.indices, book.centers)
                books[name] = Codebook(centers, book.indices, book.value_shape)
    for name in GROUPS:
        raw, book = values[name], books[name]
        replacement = book.centers[book.indices.long()].reshape_as(raw)
        result[name] = raw + (replacement - raw).detach()
    return result


def _render_view(base, tensors):
    """Do not wrap expanded tensors in Parameter: that would sever STE gradients."""
    view = copy.copy(base)
    for name, field in FIELDS.items():
        setattr(view, field, tensors[name])
        if hasattr(view, field + "_q"):
            setattr(view, field + "_q", tensors[name].new_empty((0,)))
    xyz = tensors["xyz"]
    view.optimizer = None
    view._mask = xyz.new_ones((len(xyz), 1))
    view.max_radii2D = xyz.new_zeros(len(xyz))
    view.xyz_gradient_accum = xyz.new_zeros((len(xyz), 1))
    view.denom = xyz.new_zeros((len(xyz), 1))
    return view


def _options(backend, overrides):
    parser = argparse.ArgumentParser(add_help=False)
    group = backend.arguments.OptimizationParams(parser)
    options = group.extract(parser.parse_args([]))
    for name, value in overrides.items():
        if not hasattr(options, name):
            raise ValueError("Unknown QAT optimizer option: " + name)
        setattr(options, name, value)
    if not math.isfinite(float(options.lambda_dssim)) or not 0 <= options.lambda_dssim <= 1:
        raise ValueError("lambda_dssim must lie in [0, 1]")
    return options


def _validate_basis(basis):
    if set(basis) != set(FIELDS):
        raise ValueError("QAT requires exactly the six raw Gaussian tensor fields")
    xyz = basis["xyz"]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("Expected nonempty N x 3 xyz")
    n = len(xyz)
    expected = {"features_dc": (n, 1, 3), "scaling": (n, 3),
                "rotation": (n, 4), "opacity": (n, 1)}
    for name, value in basis.items():
        if value.device != xyz.device or value.dtype != torch.float32:
            raise ValueError("QAT basis must use float32 tensors on one device")
        if value.shape[0] != n or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid point count or nonfinite values in " + name)
        if name in expected and tuple(value.shape) != expected[name]:
            raise ValueError("Invalid shape for " + name)
    rest = basis["features_rest"]
    if rest.ndim != 3 or rest.shape[1] == 0 or rest.shape[2] != 3:
        raise ValueError("Four-group QAT requires nonempty N x SH-rest x 3 coefficients")
    return n, xyz.device


def fit_qat(compact, basis_tensors, loaded, config, output_dir):
    """Return a trained four-codebook CompactScene without changing input objects.

    ``config`` is appearance.qat plus canonical ``codebook_sizes``. Raw log-scale
    and opacity logits are optimized alongside xyz and the four VQ attributes.
    Fixed template memberships and instance transforms are expanded every step.
    """
    config = dict(config or {})
    n, device = _validate_basis(basis_tensors)
    steps = _positive_integer(config.get("steps", 10000), "steps")
    start = _positive_integer(config.get("virtual_start", 20000), "virtual_start", 0)
    seed = _positive_integer(config.get("seed", 0), "seed", 0)
    requested = {name: _positive_integer(config.get("codebook_sizes", {}).get(name, 4096),
                                         "codebook_size." + name) for name in GROUPS}
    sizes = {name: min(value, n) for name, value in requested.items()}
    backend = loaded["backend"]
    cameras = list(loaded["scene"].getTrainCameras())
    if not cameras:
        raise ValueError("QAT requires at least one training camera")
    options = _options(backend, config.get("optimizer", {}))
    end = start + steps
    schedule_end = max(30000, int(getattr(options, "iterations", 30000)), end)
    base = copy.copy(loaded["gaussians"])
    for name, field in FIELDS.items():
        setattr(base, field, torch.nn.Parameter(basis_tensors[name].detach().clone()))
    base.spatial_lr_scale = float(loaded["scene"].cameras_extent)
    if not math.isfinite(base.spatial_lr_scale) or base.spatial_lr_scale <= 0:
        raise ValueError("Camera extent must be finite and positive")
    topology = copy.deepcopy(compact)
    topology.metadata["parameter_space"] = "raw"
    background = torch.tensor([1., 1., 1.] if loaded["dataset"].white_background else
                              [0., 0., 0.], device=device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info_path, log_path = out / "qat_training_info.json", out / "qat_training.jsonl"
    if info_path.exists() or log_path.exists():
        raise FileExistsError("QAT logs already exist; use a fresh output directory")
    info = dict(status="running", algorithm="four-group CompGS-style STE QAT",
                implementation="PAHC KMeans adaptation; not bitwise native CUDA reproduction",
                base_points=n, camera_count=len(cameras), seed=seed,
                requested_steps=steps, completed_steps=0, optimizer_steps=0,
                virtual_first=start + 1, virtual_last=end, schedule_end=schedule_end,
                requested_capacities=requested, capacities=sizes,
                assignment_schedule="100 steps; 5000 when iteration > schedule_end - 5000",
                last_assignment_iteration=None, final_optimizer_step_skipped=True,
                final_quantization_refreshed=False, opacity_quantized=False,
                lowbit_simulated=False, pruning=False, densification=False,
                gradient_first_nonzero_iteration={name: None for name in GROUPS},
                optimizer_options=vars(options), spatial_lr_scale=base.spatial_lr_scale)
    def save_info():
        info_path.write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    save_info()
    tick = time.perf_counter()
    books, stack = {}, []
    camera_rng = random.Random(seed)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with log_path.open("x", encoding="utf-8") as log:
        def record(row):
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
        record(dict(info, event="configuration"))
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
                if cuda_devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                base.training_setup(options)
                base.optimizer.zero_grad(set_to_none=True)
                values = {name: getattr(base, field) for name, field in FIELDS.items()}
                identities = {name: id(value) for name, value in values.items()}
                for step, iteration in enumerate(range(start + 1, end + 1), 1):
                    if any(id(getattr(base, FIELDS[name])) != identities[name] or len(value) != n
                           for name, value in values.items()):
                        raise RuntimeError("QAT changed basis topology or parameter identity")
                    base.update_learning_rate(iteration)
                    if iteration % 1000 == 0:
                        base.oneupSHdegree()
                    frequency = 5000 if iteration > schedule_end - 5000 else 100
                    assign = step == 1 or iteration % frequency == 1
                    quantized = _quantized_forward(values, books, sizes, assign)
                    if assign:
                        info["last_assignment_iteration"] = iteration
                    if not stack:
                        stack = list(cameras)
                    camera = stack.pop(camera_rng.randrange(len(stack)))
                    tensors = expand_compact_basis(topology, quantized)
                    view = _render_view(base, tensors)
                    image = backend.render(camera, view, loaded["pipe"], background)["render"]
                    target = camera.original_image.to(device=device, dtype=image.dtype)
                    if image.shape != target.shape:
                        raise ValueError("Rendered and target image shapes differ")
                    l1 = backend.l1_loss(image, target)
                    similarity = backend.ssim(image, target)
                    loss = (1. - options.lambda_dssim) * l1 + options.lambda_dssim * (1. - similarity)
                    if not bool(torch.isfinite(loss).all()):
                        raise FloatingPointError("Nonfinite QAT loss")
                    loss.backward()
                    gradients = {}
                    for name, value in values.items():
                        grad = value.grad
                        finite = grad is not None and bool(torch.isfinite(grad).all())
                        if not finite:
                            raise FloatingPointError("Missing or nonfinite gradient for " + name)
                        gradients[name] = float(grad.detach().abs().max())
                        if name in GROUPS and gradients[name] > 0 and info["gradient_first_nonzero_iteration"][name] is None:
                            info["gradient_first_nonzero_iteration"][name] = iteration
                    applied = iteration < end
                    if applied:
                        base.optimizer.step()
                        base.optimizer.zero_grad(set_to_none=True)
                        info["optimizer_steps"] += 1
                    info["completed_steps"] = step
                    record(dict(event="iteration", iteration=iteration, local_step=step,
                                camera=str(getattr(camera, "image_name", "")), loss=float(loss.detach()),
                                l1=float(l1.detach()), ssim=float(similarity.detach()),
                                assignment=assign, assignment_frequency=frequency,
                                optimizer_step=applied, gradient_max_abs=gradients))
                missing = [name for name, iteration in info["gradient_first_nonzero_iteration"].items() if iteration is None]
                if missing:
                    raise RuntimeError("No nonzero QAT gradient for: " + ", ".join(missing))
                with torch.no_grad():
                    _quantized_forward(values, books, sizes, False)
                _validate_basis(values)
            result = copy.deepcopy(topology)
            result.static_gaussians = {name: value.detach().cpu().clone()
                                       for name, value in values.items() if name not in GROUPS}
            result.codebooks = {name: dict(centers=book.centers.detach().cpu().clone(),
                                           indices=book.indices.detach().cpu().clone(),
                                           value_shape=book.value_shape)
                                for name, book in books.items()}
            info.update(status="complete", final_quantization_refreshed=True,
                        all_four_groups_received_gradients=True, seconds=time.perf_counter() - tick)
            result.metadata.update(qat=dict(info), parameter_space="raw",
                                   quantized_attributes=sorted(GROUPS),
                                   quantized_evaluation="decoded_shared_basis_and_instances")
            record(dict(info, event="complete"))
            save_info()
            return result
        except Exception as error:
            info.update(status="failed", error=repr(error), seconds=time.perf_counter() - tick)
            record(dict(info, event="failed"))
            save_info()
            raise
