"""The engine source as prompt material: the src/engine/ parts in the include order of src/forward_qwen.cu (the unity
build), as a file list or as one text (~1.6 MB of CUDA). Needle haystacks, prefill prompts and the spec-decode benches
draw their long code documents from it."""

from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def engine_source_files(repo: str = REPO) -> list[str]:
    src = os.path.join(repo, "src")
    def visit(path, ancestors=()):
        if path in ancestors:
            raise ValueError("Cyclic engine include")
        parts = []
        with open(path) as f:
            for line in f:
                match = re.match(r'#include "(?:engine/)?([\w.]+\.cuh)"', line)
                if match:
                    part = os.path.join(src, 'engine', match.group(1))
                    parts.append(part)
                    parts.extend(visit(part, (*ancestors, path)))
        return parts
    return visit(os.path.join(src, "forward_qwen.cu"))


def engine_source_text(repo: str = REPO) -> str:
    parts = []
    for p in engine_source_files(repo):
        with open(p) as f:
            parts.append(f.read())
    return "".join(parts)
