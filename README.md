# HPS3DGS

Code for **Hierarchical Prototype Sharing for 3D Gaussian Splatting Compression
of Aerospace Objects**.
> The paper for this code was previously titled "Physics-Aware Hierarchical 3DGS Compression: An Efficient Representation for Aerospace Objects".

HPS3DGS exploits repeated rigid components and material-related attribute
regularities in aerospace objects to compress an unstructured 3D Gaussian
Splatting (3DGS) scene. Its hierarchical representation combines shared Gaussian
templates and instance transformations in a geometry layer with independent
attribute codebooks in an appearance layer.

Training and compression are performed offline. The stored representation is
decoded by retrieving attributes from codebooks and expanding shared templates
at their instance poses for Gaussian rendering, reducing model storage and
transmission requirements.

[Project page](https://hps3dgs.github.io/)

The repository directory, package and command names retain `pahc3dgs` / `PAHC`
for compatibility with existing workflows.

## Method Overview

1. **Geometry layer:** semantic-spatial decomposition identifies candidate
   components. Spectral matching, FPFH/RANSAC alignment and alternating
   refinement of instance transformations and Gaussian parameters establish
   template-instance correspondences for repeated structures.
2. **Appearance layer:** separate codebooks quantize SH-rest, DC, rotation and
   scale attributes. Quantization-aware training (QAT) uses codeword values in
   the forward pass and straight-through gradients to refine the continuous
   shared representation under the rendering loss.
3. **Storage and decoding:** post-training low-bit encoding stores codebooks
   and opacity logits in FP16, positions in 14 bits per coordinate, and index
   maps in ceil(log2(K)) bits per attribute group. Decoding reconstructs the
   Gaussian attributes and assembles the template instances for rendering.

The paper evaluates synthetic and real spacecraft sequences from SPARD using
the same 150 views per sequence for training and reconstruction evaluation.
These results characterize representation fidelity on the observed views.
Implementation-specific details and the scope of this integration are described
in [QAT and Low-Bit Storage](#qat-and-low-bit-storage).

## Project Layout

```text
pahc3dgs/
├── configs/pahc.yaml
├── scripts/
│   ├── run_pahc_pipeline.py
│   ├── compress.py
│   ├── decode.py
│   └── evaluate.py
├── src/
│   ├── backend.py
│   ├── data.py
│   ├── compression.py
│   ├── math_utils.py
│   ├── quantization.py
│   ├── qat.py
│   ├── lowbit.py
│   ├── codec.py
│   └── metrics.py
├── third_party/SegAnyGAussians/
└── tests/
```

## Third-Party Backend

This implementation uses SegAnyGaussians as the third-party backend for standard 3DGS,
semantic feature training, scene loading, and rendering.

```bash
git submodule update --init --recursive third_party/SegAnyGaussians
```

## Installation

This repository is installed incrementally on top of an existing
SegAnyGaussians (SAGA) environment. First install SAGA, PyTorch, torchvision,
and its CUDA rasterization extensions by following the upstream SAGA
instructions. Then install the project-specific dependencies:

```bash
conda activate sacgs
pip install -r requirements.txt
pip install -e .
```

The `requirements.txt` file intentionally does not install or replace PyTorch,
torchvision, or the SAGA CUDA extensions. This avoids changing the GPU stack in
an environment where SAGA already runs correctly. The current setup has been
verified with Python 3.8.20, PyTorch 2.4.1+cu121, and torchvision 0.19.1+cu121
in the `sacgs` environment.

## Tests

```bash
python -m unittest discover -s tests
```

## Full Pipeline

Use the unified entry point for training, semantic feature extraction, feature
Gaussian training, and hierarchical prototype-sharing compression:

```bash
conda activate sacgs
cd /path/to/pahc3dgs
mkdir -p outputs logs

set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python scripts/run_pahc_pipeline.py \
  -s /path/to/scene \
  -m outputs/scene_3dgs \
  -o outputs/scene.pahc.pt \
  --saga-root /path/to/SegAnyGaussians \
  --sam-checkpoint-path /path/to/sam_vit_h_4b8939.pth \
  2>&1 | tee logs/scene_full_pipeline.log
```

`set -o pipefail` preserves a nonzero exit status when a pipeline stage fails,
while `tee` keeps the complete console output in a log file. Use a new model
directory for each from-scratch run because SAGA writes checkpoints and
configuration files into that directory.

To compress an existing trained SAGA scene, skip the first three stages:

```bash
python scripts/run_pahc_pipeline.py \
  -s /path/to/scene \
  -m /path/to/SegAnyGaussians/output/scene \
  --saga-root /path/to/SegAnyGaussians \
  --skip-train --skip-mask --skip-feature
```

With the default configuration, the compression stage performs semantic-spatial
clustering, spectral and semantic matching, FPFH/RANSAC initialization, pose
rejection, seven alternating pose/parameter refinement cycles, target removal,
four-group QAT and low-bit storage. Rendering evaluation compares the
unquantized representation with the decoded stored model, including its instances.

## Compact Representation

`src.codec.CompactScene` stores:

- `static_gaussians`: the unique Gaussian basis fields retained outside the
  codebooks, including positions and opacity logits in the default configuration.
- `templates`: index sets identifying shared templates within the unique basis.
- `instances`: template and instance ids, rotation, translation, center, and
  optional scale used to reconstruct repeated components.
- `codebooks`: attribute-specific centers, index maps and tensor shapes.
- `metadata` and `version`: representation statistics, training/evaluation
  records, parameter conventions and the payload version.

Low-bit packing is applied when saving the payload; loading restores the tensor
representation used by the decoder.

## Acknowledgements

This implementation builds on common 3DGS research infrastructure. It uses
[SegAnyGaussians](https://github.com/Jumpat/SegAnyGAussians) as a third-party backend and reorganizes the KMeans appearance
quantization ideas from [CompGS](https://github.com/UCDvision/compact3d) into the appearance layer.

## QAT and Low-Bit Storage

The default `configs/pahc.yaml` enables 10,000 scheduled QAT steps (9,999
optimizer updates) after geometry compression. `src/qat.py` adapts the verified
Geo33/CompGS schedule to the existing PAHC model: SH-rest, DC, raw rotation and
log-scale use codeword forward values with straight-through gradients. Centers
are recomputed outside autograd; assignments refresh every 100 virtual steps,
then every 5,000 in the last 5,000 steps. Continuous xyz and opacity logits also
optimize, while template membership, instance transforms and point count stay
fixed. QAT uses the SAGA L1 + D-SSIM loss and does not simulate storage rounding.
The actual codebook capacity is min(configured K, basis point count) and is logged.

`src/lowbit.py` stores codebook centers and opacity logits in FP16, xyz in 14
bits per coordinate using a global per-axis bounding box, and each attribute's
indices in ceil(log2(K)) bits. The existing `.pt` container and public load/decode
functions remain in use; older payloads still load. Decoding restores FP32 tensors
for rendering. Reported rendering metrics now use the saved, reloaded package
including all reconstructed instances. `scripts/evaluate.py` counts complete file
bytes, with the original size remaining an FP32 parameter estimate.

Use the existing `--config` option on either compression entry point. For the
paired ablations, copy the default config and change only:

| Variant | `appearance.qat.enabled` | `codec.lowbit` |
|---|---|---|
| Full | `true` | `true` |
| w/o QAT (four-group PTQ) | `false` | `true` |
| w/o Low-bit | `true` | `false` |

For a strictly paired QAT/PTQ comparison, reuse the same continuous geometry
state and camera protocol: toggling the config while rerunning geometry alone
does not guarantee matched inputs.

For a strictly paired storage comparison, encode the same trained CompactScene
with `save_compact_scene(scene, path, lowbit=False)` and `lowbit=True`; do not
retrain QAT for each precision. QAT logs are saved beside the output in its `.qat`
directory; use a fresh output name for each training run. Older configs that omit
both switches retain PTQ and ordinary storage; their attribute list is respected.

The reference implementation is the `ours-sparsity-all5-20260923` HPS3DGS
experiment (`geo33_appearance.py` and `geo33_storage_codec.py`). This integration
reuses PAHC's KMeans, geometry and instance conventions (including its existing
zero SH-rest for generated instances), and uses a PAHC `.pt` payload rather than
the reference ZIP format. It does not import the external experiment worktree,
add its sparsity stage, or claim reproduction of the paper's numerical results.
