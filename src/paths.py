"""Output locations. Setting LEAF_VARIANT=<name> redirects the pipeline to a parallel set of files
(data/metadata_<name>.csv, artifacts/variant_<name>/...) so a sensitivity experiment never overwrites the primary results.
With LEAF_VARIANT unset everything is exactly as before (data/metadata.csv, artifacts/...)."""
import os
from pathlib import Path

VARIANT = os.environ.get("LEAF_VARIANT", "").strip()
ART = Path("artifacts") / (f"variant_{VARIANT}" if VARIANT else "")
META = Path(f"data/metadata_{VARIANT}.csv" if VARIANT else "data/metadata.csv")


def ensure(*subdirs):
    for s in subdirs:
        (ART / s).mkdir(parents=True, exist_ok=True)
