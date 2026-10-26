"""Check that data/raw contains exactly the images listed in the split manifest (SHA-256 match).
Usage: python -m src.verify_data   (exit code 0 = all files present and identical)"""
import sys

from src.data import file_sha256, load_metadata

UPSTREAM = "https://github.com/spMohanty/PlantVillage-Dataset @ 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a (raw/color)"


def main():
    df = load_metadata("data/metadata.csv")
    missing, changed = [], []
    for r in df.itertuples():
        try:
            if file_sha256(r.path) != r.sha256:
                changed.append(r.path)
        except FileNotFoundError:
            missing.append(r.path)
    print(f"{len(df) - len(missing) - len(changed)}/{len(df)} files match the manifest; missing {len(missing)}, different {len(changed)}")
    if missing or changed:
        print("Re-download from", UPSTREAM)
    sys.exit(1 if missing or changed else 0)


if __name__ == "__main__":
    main()
