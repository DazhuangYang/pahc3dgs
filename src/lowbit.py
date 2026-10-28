"""Low-bit storage adapted to the PAHC torch payload (not a Geo33 ZIP).

Uses the field precisions and big-endian bit packing of the verified source:
/disk3/ydz/code-worktrees/HPS3DGS/ours-sparsity-all5-20260923/
third_party/SegAnyGAussians/geo33_storage_codec.py
SHA256: 2bfa612fa9993218c78fb48e8f8d4b085e7df0c5bc5a722925f301ff37f20400
Only serialization changes: FP16 centers/opacity, per-group ceil(log2 K)
indices, and global-bbox 14-bit xyz. Training does not simulate these errors.
The existing PT container, template relations and instance poses are retained.
"""

import math

import numpy as np
import torch

LOWBIT_FORMAT = "pahc-lowbit-v1"
XYZ_BITS = 14


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("Invalid " + name)
    return value


def _float_tensor(value, name, dtype=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(name + " must be a floating tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(name + " has the wrong storage dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " contains nonfinite values")
    return value.detach()


def _half(value, name):
    value = _float_tensor(value, name).to(dtype=torch.float16)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " overflows FP16")
    return value


def _pack(values, width):
    """Store actual bits, with only the final byte padded with zeroes."""
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    if width == 0:
        if np.any(values):
            raise ValueError("Nonzero value in a zero-bit field")
        return torch.empty(0, dtype=torch.uint8)
    if not 0 < width <= 32 or np.any(values >= (1 << width)):
        raise ValueError("Value does not fit the declared width")
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8)
    return torch.from_numpy(np.packbits(bits.reshape(-1), bitorder="big"))


def _unpack(data, count, width):
    count = _integer(count, "field count")
    width = _integer(width, "field width")
    if width > 32:
        raise ValueError("Field width exceeds 32 bits")
    total_bits = count * width
    if (not isinstance(data, torch.Tensor) or data.dtype != torch.uint8 or
            data.ndim != 1 or data.numel() != (total_bits + 7) // 8):
        raise ValueError("Bitstream length or dtype mismatch")
    bits = np.unpackbits(data.detach().cpu().numpy(), bitorder="big")
    if np.any(bits[total_bits:]):
        raise ValueError("Nonzero bitstream padding")
    if width == 0:
        return np.zeros(count, dtype=np.uint32)
    weights = np.left_shift(np.uint64(1), np.arange(width - 1, -1, -1, dtype=np.uint64))
    return (bits[:total_bits].reshape(count, width).astype(np.uint64) @ weights).astype(np.uint32)


def _shape(codebook, centers):
    if centers.ndim != 2 or centers.shape[0] < 1:
        raise ValueError("Codebook centers must be nonempty K x D tensors")
    shape = tuple(codebook.get("value_shape", (centers.shape[1],)))
    for dim in shape:
        _integer(dim, "attribute dimension")
    if math.prod(shape) != centers.shape[1]:
        raise ValueError("Codebook value_shape disagrees with centers")
    return shape


def encode_payload(payload):
    """Return a low-bit payload without mutating its continuous/VQ source."""
    if "storage" in payload:
        raise ValueError("Payload is already storage-encoded")
    result = dict(payload)
    static = dict(payload.get("static_gaussians", {}))
    xyz = _float_tensor(static.get("xyz"), "xyz")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("xyz must be a nonempty N x 3 tensor")
    _float_tensor(xyz.float(), "xyz as FP32")
    count = int(xyz.shape[0])
    array = xyz.cpu().double().numpy()
    minimum, maximum = array.min(0), array.max(0)
    span = maximum - minimum
    levels = (1 << XYZ_BITS) - 1
    normalized = np.divide(array - minimum, span, out=np.zeros_like(array), where=span > 0)
    indices = np.rint(normalized * levels).clip(0, levels).astype(np.uint32)
    static["xyz"] = {
        "bits": _pack(indices, XYZ_BITS), "count": count,
        "bbox_min": torch.from_numpy(minimum.copy()),
        "bbox_max": torch.from_numpy(maximum.copy()),
    }
    if "opacity" in static:
        opacity = _float_tensor(static["opacity"], "opacity")
        if tuple(opacity.shape) != (count, 1):
            raise ValueError("opacity must be N x 1 logits")
        static["opacity"] = _half(opacity, "opacity")
    encoded = {}
    for name, original in payload.get("codebooks", {}).items():
        centers = _float_tensor(original["centers"], name + " centers")
        shape = _shape(original, centers)
        ids = original["indices"]
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.numel() != count or
                ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("Codebook indices must be one integer per basis point")
        k = int(centers.shape[0])
        if bool((ids < 0).any()) or bool((ids >= k).any()):
            raise ValueError("Codebook index is out of range")
        width = (k - 1).bit_length()
        book = dict(original)
        del book["indices"]
        book.update(centers=_half(centers, name + " centers"), value_shape=shape,
                    packed_indices=_pack(ids.detach().cpu().numpy(), width),
                    index_bits=width, count=count)
        encoded[name] = book
    result.update(static_gaussians=static, codebooks=encoded, storage={
        "format": LOWBIT_FORMAT, "xyz_bits": XYZ_BITS,
        "centers_dtype": "float16", "opacity_dtype": "float16",
        "index_bits": "ceil(log2(K))", "bitorder": "big",
    })
    return result


def decode_payload(payload):
    """Restore FP32 tensors and integer indices, accepting legacy payloads."""
    if "storage" not in payload:
        return payload
    spec = payload["storage"]
    expected = {"format": LOWBIT_FORMAT, "xyz_bits": XYZ_BITS,
                "centers_dtype": "float16", "opacity_dtype": "float16",
                "index_bits": "ceil(log2(K))", "bitorder": "big"}
    if spec != expected:
        raise ValueError("Unsupported low-bit storage settings")
    result = dict(payload)
    del result["storage"]
    static = dict(payload.get("static_gaussians", {}))
    geometry = static.get("xyz")
    if not isinstance(geometry, dict):
        raise ValueError("Missing packed xyz")
    count = _integer(geometry.get("count"), "basis count", 1)
    minimum = _float_tensor(geometry["bbox_min"], "bbox_min", torch.float64)
    maximum = _float_tensor(geometry["bbox_max"], "bbox_max", torch.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or bool((maximum < minimum).any()):
        raise ValueError("Invalid xyz bounds")
    values = _unpack(geometry["bits"], count * 3, XYZ_BITS).reshape(count, 3)
    lo, hi = minimum.cpu().numpy(), maximum.cpu().numpy()
    restored = lo + values.astype(np.float64) * ((hi - lo) / ((1 << XYZ_BITS) - 1))
    xyz = torch.from_numpy(restored).float().to(geometry["bits"].device)
    _float_tensor(xyz, "decoded xyz")
    static["xyz"] = xyz
    if "opacity" in static:
        opacity = _float_tensor(static["opacity"], "opacity", torch.float16)
        if tuple(opacity.shape) != (count, 1):
            raise ValueError("opacity shape disagrees with basis count")
        static["opacity"] = opacity.float()
    decoded = {}
    for name, original in payload.get("codebooks", {}).items():
        centers = _float_tensor(original["centers"], name + " centers", torch.float16)
        shape = _shape(original, centers)
        k = int(centers.shape[0])
        width = _integer(original.get("index_bits"), "index width")
        book_count = _integer(original.get("count"), "codebook count", 1)
        if width != (k - 1).bit_length() or book_count != count or "indices" in original:
            raise ValueError("Invalid codebook layout")
        ids = _unpack(original["packed_indices"], count, width)
        if np.any(ids >= k):
            raise ValueError("Decoded index exceeds codebook capacity")
        book = dict(original)
        for key in ("packed_indices", "index_bits", "count"):
            del book[key]
        book.update(centers=centers.float(), value_shape=shape,
                    indices=torch.from_numpy(ids.astype(np.int64)).to(centers.device))
        decoded[name] = book
    result.update(static_gaussians=static, codebooks=decoded)
    return result
