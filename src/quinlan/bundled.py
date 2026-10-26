"""Locations and provenance for the regular source files shipped with Quinlan."""

import hashlib
import json
import os
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "_vendor"


def source_dir(component="qwentin"):
    if component not in {"qwentin", "exllamav3"}:
        raise ValueError("Unknown bundled component")
    source = VENDOR / component
    if not (source / "LICENSE").is_file():
        raise FileNotFoundError(f"Bundled {component} source is missing; reinstall the complete Quinlan package")
    return source


def verify_sources():
    manifest = json.loads((VENDOR / "SOURCES.json").read_text())
    counts = {}
    for component, item in manifest.items():
        root = source_dir(component)
        expected = item["files"]
        for name, entry in expected.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid source manifest path")
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Bundled source integrity check failed: {component}/{name}")
        counts[component] = len(expected)
    return counts


def native_fingerprint(source=None):
    root = Path(source) if source is not None else source_dir()
    paths = [root / "CMakeLists.txt"]
    paths.extend(p for p in sorted((root / "src").rglob("*")) if p.is_file())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def build_dir(source=None):
    explicit = os.environ.get("QUINLAN_BUILD_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    root = Path(source).resolve() if source is not None else source_dir()
    # CMake caches an absolute source directory. Identical sources installed in
    # two environments must not share a CMake build directory.
    identity = native_fingerprint(root) + "\0" + str(root)
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return cache / "quinlan" / "native" / key


def library_path(source=None):
    return build_dir(source) / "libforward_qwen.so"
