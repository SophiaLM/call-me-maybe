import os
import tempfile

from src.cli import parse_args, resolve_paths


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.input_path is None
        assert args.output_path is None

    def test_custom_input(self):
        args = parse_args(["--input", "/custom/path/tests.json"])
        assert args.input_path == "/custom/path/tests.json"

    def test_custom_input_short(self):
        args = parse_args(["-i", "/custom/path/tests.json"])
        assert args.input_path == "/custom/path/tests.json"

    def test_custom_output(self):
        args = parse_args(["--output", "/custom/output.json"])
        assert args.output_path == "/custom/output.json"

    def test_custom_output_short(self):
        args = parse_args(["-o", "/custom/output.json"])
        assert args.output_path == "/custom/output.json"

    def test_both_custom(self):
        args = parse_args(["-i", "/in.json", "-o", "/out.json"])
        assert args.input_path == "/in.json"
        assert args.output_path == "/out.json"


class TestResolvePaths:
    def test_default_paths(self):
        args = parse_args([])
        input_dir, output_path = resolve_paths(args)
        assert input_dir == os.path.join("data", "input")
        assert output_path == os.path.join("data", "output", "function_calling_results.json")

    def test_custom_input_derives_dir(self):
        args = parse_args(["--input", "/some/dir/tests.json"])
        input_dir, output_path = resolve_paths(args)
        assert input_dir == "/some/dir"
        assert output_path == os.path.join("data", "output", "function_calling_results.json")

    def test_custom_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "results.json")
            args = parse_args(["--output", output])
            input_dir, output_path = resolve_paths(args)
            assert input_dir == os.path.join("data", "input")
            assert output_path == output
            assert os.path.isdir(tmpdir)

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "new_dir", "results.json")
            args = parse_args(["--output", output])
            input_dir, output_path = resolve_paths(args)
            assert output_path == output
            assert os.path.isdir(os.path.join(tmpdir, "new_dir"))
