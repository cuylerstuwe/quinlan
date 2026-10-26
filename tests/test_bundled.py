"""Bundled build/runtime paths work independently of upstream Git checkouts."""
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from quinlan import build, bundled, convert, server


class BundledTests(unittest.TestCase):
    def test_manifest_and_required_runtime_assets(self):
        counts = bundled.verify_sources()
        self.assertGreater(counts["qwentin"], 50)
        self.assertGreater(counts["exllamav3"], 600)
        self.assertEqual((bundled.source_dir() / "tools/codebooks/lloyd_kv12a.f32").stat().st_size, 16384)
        self.assertEqual((bundled.source_dir() / "src/engine/quinlan_sampling.cuh").read_bytes(),
                         (Path(server.__file__).parent / "native/sampling.cuh").read_bytes())
        for component in counts:
            self.assertFalse((bundled.source_dir(component) / ".git").exists())

    def test_scheduler_imports_from_included_source(self):
        root = bundled.source_dir()
        with mock.patch.object(sys, "path", [str(root), *sys.path]):
            module = importlib.import_module("tools.serve_batched")
        self.assertEqual(Path(module.__file__).resolve(), root / "tools/serve_batched.py")
        self.assertTrue(callable(module.BatchedEngine.submit))
        self.assertIn("sampling", module.Request.__slots__)

    def test_server_defaults_to_bundled_source_and_built_library(self):
        args = server.parse_args(["--model-dir", "model", "--tqf", "model.tqf", "--api-key-file", "key"])
        self.assertEqual(Path(args.source), bundled.source_dir())
        self.assertEqual(Path(args.lib), bundled.library_path())

    def test_build_uses_only_bundled_source_and_chosen_toolchain(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(build.subprocess, "run") as run:
            with mock.patch("sys.stdout", new=io.StringIO()):
                build.main(["--build-dir", directory, "--cuda-compiler", "/toolkit/nvcc", "--jobs", "1"])
            configure = run.call_args_list[0].args[0]
            self.assertEqual(configure[:3], ["cmake", "-S", str(bundled.source_dir())])
            self.assertIn("-DCMAKE_CUDA_COMPILER=/toolkit/nvcc", configure)
            self.assertEqual(run.call_args_list[1].args[0], ["cmake", "--build", str(Path(directory).resolve()), "--parallel", "1"])

    def test_native_source_edit_changes_build_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "src").mkdir()
            (source / "CMakeLists.txt").write_text("project(test)")
            header = source / "src/test.cuh"
            header.write_text("first")
            before = bundled.native_fingerprint(source)
            header.write_text("second")
            self.assertNotEqual(bundled.native_fingerprint(source), before)

    def test_moving_identical_sources_uses_a_fresh_cmake_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            (first / "src").mkdir(parents=True)
            (first / "CMakeLists.txt").write_text("project(test)")
            second = Path(directory) / "second"
            shutil.copytree(first, second)
            with mock.patch.dict(os.environ, {"QUINLAN_BUILD_DIR":"", "XDG_CACHE_HOME":directory}):
                self.assertEqual(bundled.native_fingerprint(first), bundled.native_fingerprint(second))
                self.assertNotEqual(bundled.build_dir(first), bundled.build_dir(second))

    def test_converter_dispatches_to_included_sources(self):
        for command, (component, script) in convert.SCRIPTS.items():
            with mock.patch.object(convert.subprocess, "call", return_value=0) as call:
                with self.assertRaises(SystemExit) as done:
                    convert.main([command, "--help"])
            self.assertEqual(done.exception.code, 0)
            self.assertEqual(call.call_args.args[0][1:], [str(bundled.source_dir(component) / script), "--help"])
            paths = call.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
            self.assertEqual(paths[:2], [str(bundled.source_dir("exllamav3")), str(bundled.source_dir())])
            if command == "export":
                self.assertEqual(call.call_args.kwargs["env"]["TQ_FORWARD_LIB"], os.environ.get("TQ_FORWARD_LIB", str(bundled.library_path())))

    def test_quantizer_recipe_preset_is_local(self):
        with mock.patch.object(convert.subprocess, "call", return_value=0) as call:
            with self.assertRaises(SystemExit):
                convert.main(["quantize", "--recipe", "opt5.75", "-i", "model", "-cd", "calibration.safetensors"])
        self.assertEqual(call.call_args.args[0][2:4], ["--recipe", str(bundled.source_dir() / "tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml")])

    def test_quantizer_requires_explicit_calibration_and_bundles_no_datasets(self):
        with mock.patch.object(convert.subprocess, "call") as call, mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                convert.main(["quantize", "-i", "model"])
        self.assertEqual(error.exception.code, 2)
        call.assert_not_called()
        self.assertFalse(list(bundled.VENDOR.rglob("*.utf8")))
        self.assertFalse(list(bundled.VENDOR.rglob("*.safetensors")))
