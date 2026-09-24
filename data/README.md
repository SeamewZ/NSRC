# Released input snapshot

This directory contains the source modalities used by the cryptocurrency event pipeline: 5-minute OHLCV bars, event/news records, core celebrity posts, and asset-related posts. Auxiliary candlestick images and legacy prediction dumps are not part of the NSRC release.

Run `python scripts/summarize_bundle.py --bundle-root data/celebcrypto` to inspect the snapshot, then use the data preparation commands documented in the root README. Macro-event and equity-price inputs are obtained through the dedicated preparation scripts and are not represented by these crypto files.
