"""Run the included quantizer, draft-head builder, TQF exporter, or inspector."""

import argparse
import os
import subprocess
import sys

from .bundled import library_path, source_dir

SCRIPTS = {
    "quantize": ("exllamav3", "convert.py"),
    "draft": ("qwentin", "tools/draft_head_build.py"),
    "export": ("qwentin", "tools/convert_qwen_tqf.py"),
    "inspect": ("qwentin", "tools/inspect_tqf.py"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=SCRIPTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments forwarded to the selected tool")
    args = parser.parse_args(argv)
    component, script = SCRIPTS[args.command]
    arguments = list(args.arguments)
    if args.command == "quantize":
        has_calibration = any(value in {"-cd", "--cal_data", "--resume"} or value.startswith("--cal_data=")
                              for value in arguments)
        if not has_calibration and not any(value in {"-h", "--help"} for value in arguments):
            parser.error("quantize requires -cd/--cal_data with your calibration token file; calibration datasets are not distributed")
        recipe = str(source_dir() / "tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml")
        for index, value in enumerate(arguments):
            if value in {"-rcp", "--recipe"} and index + 1 < len(arguments) and arguments[index + 1] == "opt5.75":
                arguments[index + 1] = recipe
            elif value == "--recipe=opt5.75":
                arguments[index] = "--recipe=" + recipe
    env = os.environ.copy()
    if args.command == "export":
        env.setdefault("TQ_FORWARD_LIB", str(library_path()))
    roots = [str(source_dir("exllamav3")), str(source_dir("qwentin"))]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    raise SystemExit(subprocess.call([sys.executable, str(source_dir(component) / script), *arguments], env=env))


if __name__ == "__main__":
    main()
