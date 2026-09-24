#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from celebcrypto.data.bundled import summarize_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact summary of assets/dataset.")
    parser.add_argument("--bundle-root", default="assets/dataset")
    args = parser.parse_args()
    print(json.dumps(summarize_bundle(args.bundle_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

