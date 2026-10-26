"""Build the included CUDA engine without fetching or patching another repository."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from .bundled import build_dir, source_dir, verify_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="optional alternative engine source; default is bundled")
    parser.add_argument("--build-dir", type=Path, help="default: source-keyed user cache, or QUINLAN_BUILD_DIR")
    parser.add_argument("--cuda-compiler", help="nvcc path; default CUDACXX or nvcc on PATH")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true", help="check bundled source hashes without CUDA")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.verify_only:
        print(json.dumps({"verified_files": verify_sources()}))
        return
    source = args.source.resolve() if args.source else source_dir()
    destination = args.build_dir.resolve() if args.build_dir else build_dir(source)
    compiler = args.cuda_compiler or os.environ.get("CUDACXX") or shutil.which("nvcc")
    if not compiler:
        parser.error("CUDA Toolkit 13.1 nvcc is required; set --cuda-compiler or CUDACXX")
    subprocess.run(["cmake", "-S", str(source), "-B", str(destination),
                    f"-DCMAKE_CUDA_COMPILER={compiler}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(destination), "--parallel", str(args.jobs)], check=True)
    print(json.dumps({"library": str(destination / "libforward_qwen.so")}))


if __name__ == "__main__":
    main()
