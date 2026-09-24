#!/usr/bin/env python3
from __future__ import annotations

import argparse

from celebcrypto.data.bundled import export_standardized_bundle
from celebcrypto.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the bundled assets/dataset snapshot into standardized CelebCrypto tables."
    )
    parser.add_argument("--bundle-root", default="assets/dataset")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    logger = get_logger("prepare_from_bundle")
    manifest = export_standardized_bundle(args.bundle_root, args.output_dir)
    logger.info("Converted bundled dataset from %s", args.bundle_root)
    for key, value in manifest.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()

