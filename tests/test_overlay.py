"""Pinned-source patch preflight, repeatability and tool-parser provenance."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from quinlan.patch import apply, QWENTIN_REVISION
from quinlan import tool_parser


@unittest.skipUnless(os.environ.get("QWENTIN_SOURCE"), "set QWENTIN_SOURCE for overlay checks")
class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "qwentin"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout",
                        os.environ["QWENTIN_SOURCE"], str(self.source)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.source), "checkout", "--quiet", "--detach", QWENTIN_REVISION],
                       check=True, capture_output=True)

    def test_apply_and_repeat(self):
        self.assertTrue(apply(self.source))
        self.assertFalse(apply(self.source))
        self.assertIn("quinlan_spec_verify", (self.source / "src/engine/paged_spec.cuh").read_text())
        self.assertTrue((self.source / "src/engine/quinlan_sampling.cuh").is_file())
        compile((self.source / "tools/serve_batched.py").read_text(), "serve_batched.py", "exec")

    def test_conflict_does_not_partially_modify_checkout(self):
        target = self.source / "src/engine/paged_t4.cuh"
        target.write_text(target.read_text().replace("paged_free_all(); return 0;", "return -1;"))
        before = (self.source / "src/forward_qwen.cu").read_bytes()
        with self.assertRaises(ValueError):
            apply(self.source)
        self.assertEqual((self.source / "src/forward_qwen.cu").read_bytes(), before)
        self.assertFalse((self.source / "src/engine/quinlan_sampling.cuh").exists())

    def test_wrong_revision_is_rejected(self):
        # A different object ID also works with shallow clones and creates no commits.
        tree = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD^{tree}"], text=True)
        (self.source / ".git/HEAD").write_text(tree)
        with self.assertRaises(ValueError):
            apply(self.source)
        self.assertFalse((self.source / "src/engine/quinlan_sampling.cuh").exists())

    def test_tool_parser_matches_pinned_upstream(self):
        def functions(path):
            tree = ast.parse(Path(path).read_text())
            return [ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in {"_coerce_arg", "parse_tool_calls"}]
        self.assertEqual(functions(self.source / "tools/serve_batched.py"), functions(tool_parser.__file__))
