#!/usr/bin/env python3
"""Compute PAHC-3DGS compression statistics."""


import argparse
import json
from pathlib import Path

from src.codec import load_compact_scene
from src.metrics import compression_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-scene", type=Path, required=True)
    parser.add_argument("--original-count", type=int, required=True)
    parser.add_argument("--final-count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene = load_compact_scene(args.compact_scene)
    final_count = args.final_count
    if final_count is None:
        xyz = scene.static_gaussians.get("xyz")
        if xyz is None:
            xyz = scene.templates.get("xyz")
        final_count = int(xyz.shape[0]) if xyz is not None else 0
    report = compression_report(args.original_count, final_count, final_bytes=args.compact_scene.stat().st_size)
    report["storage_basis"] = "original FP32 parameter estimate / complete saved file bytes"
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
