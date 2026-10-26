"""Overlay regressions using bundled source; no upstream checkout or network."""
import ast
from importlib.resources import files
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from quinlan.bundled import source_dir
from quinlan.patch import apply, QWENTIN_REVISION
from quinlan import tool_parser


@unittest.skipUnless(shutil.which("git"), "legacy patch utility checks need git")
class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "qwentin"
        shutil.copytree(source_dir(), self.source)
        patch = files("quinlan").joinpath("native/qwentin.patch").read_bytes()
        subprocess.run(["git", "-C", str(self.source), "apply", "--reverse", "-"],
                       input=patch, check=True, capture_output=True)
        (self.source / "src/engine/quinlan_sampling.cuh").unlink()
        # The compatibility utility validates upstream Git identity. Mock only that
        # lookup; its real patch preflight runs on source reconstructed from this repo.
        self.pin = mock.patch("quinlan.patch.subprocess.check_output", return_value=QWENTIN_REVISION + "\n")
        self.pin.start()
        self.addCleanup(self.pin.stop)

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
        with mock.patch("quinlan.patch.subprocess.check_output", return_value="0" * 40):
            with self.assertRaises(ValueError):
                apply(self.source)
        self.assertFalse((self.source / "src/engine/quinlan_sampling.cuh").exists())


class BundledParserTests(unittest.TestCase):
    def test_tool_parser_matches_bundled_engine(self):
        def functions(path):
            tree = ast.parse(Path(path).read_text())
            return [ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in {"_coerce_arg", "parse_tool_calls"}]
        self.assertEqual(functions(source_dir() / "tools/serve_batched.py"), functions(tool_parser.__file__))
