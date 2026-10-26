"""Optional pinned ExLlamaV3 diagnostic-memory override; calibration is unchanged."""
import argparse
from pathlib import Path
import subprocess

REVISION = "da68d57de61781e5b74ad68698ac4caf36408d7a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Expected the supported ExLlamaV3 revision; no edits made")
    target = args.source / "exllamav3/conversion/convert_model.py"
    text = target.read_text()
    before = "num_ref_states = 5"
    after = 'num_ref_states = int(os.environ.get("EXL3_CONVERT_DIAGNOSTIC_ROWS", "5"))'
    if after not in text:
        if text.count(before) != 1:
            raise ValueError("Unexpected converter source; no edits made")
        target.write_text(text.replace(before, after))
    print("Diagnostic-row override available; set EXL3_CONVERT_DIAGNOSTIC_ROWS=1 to use it.")


if __name__ == "__main__":
    main()
