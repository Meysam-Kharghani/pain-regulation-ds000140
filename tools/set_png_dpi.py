#!/usr/bin/env python3
"""Rewrite PNG resolution metadata without resampling image pixels."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    with Image.open(args.input) as image:
        image.save(args.output, dpi=(args.dpi, args.dpi))
    print(f"[OK] Output saved: {args.output} with {args.dpi} dpi metadata; pixels unchanged.")


if __name__ == "__main__":
    main()
