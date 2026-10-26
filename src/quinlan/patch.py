"""Apply the native overlay to a pinned Qwentin checkout, after a full preflight."""

import argparse
from importlib.resources import files
from pathlib import Path
import subprocess

QWENTIN_REVISION = "8a626374912fe1e24649403423c1120ff58d8522"


def apply(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != QWENTIN_REVISION:
        raise ValueError(f"Expected Qwentin revision {QWENTIN_REVISION}; checkout is unchanged")
    native = files("quinlan").joinpath("native")
    patch = native.joinpath("qwentin.patch").read_bytes()
    header = native.joinpath("sampling.cuh").read_bytes()
    target = source / "src/engine/quinlan_sampling.cuh"
    if target.exists() and target.read_bytes() != header:
        raise ValueError("Native sampler differs from the packaged version; checkout is unchanged")

    def git_apply(*options):
        return subprocess.run(
            ["git", "-C", str(source), "apply", *options, "-"],
            input=patch, capture_output=True, check=False,
        )

    if git_apply("--reverse", "--check").returncode == 0:
        if not target.exists():
            target.write_bytes(header)
        return False
    if git_apply("--check").returncode:
        raise ValueError("Overlay conflicts with local source changes; checkout is unchanged")
    result = git_apply()
    result.check_returncode()
    target.write_bytes(header)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Qwentin checkout at the supported revision")
    args = parser.parse_args()
    changed = apply(args.source)
    print("Quinlan overlay applied; rebuild the engine." if changed else "Quinlan overlay already applied.")


if __name__ == "__main__":
    main()
