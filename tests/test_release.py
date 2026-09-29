"""CPU-only repository checks. No torch, Ray, network, model, or GPU dependencies."""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("ours", "plain", "grpd", "opdgrpo", "exopd")


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


class DocumentationTests(unittest.TestCase):
    def test_local_document_links(self):
        documents = [ROOT / "README.md", ROOT / "README_zh.md", ROOT / "UPSTREAM.md",
                     ROOT / "LICENSE.md", ROOT / "OPDVR/README.md", *sorted((ROOT / "docs").glob("*.md"))]
        for document in documents:
            text = document.read_text()
            parser = LinkParser()
            parser.feed(text)
            links = parser.links + re.findall(r"\]\(([^)]+)\)", text)
            for link in links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                target = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
                with self.subTest(document=document.name, link=link):
                    self.assertTrue(target.is_relative_to(ROOT))
                    self.assertTrue(target.exists(), f"Missing target: {target}")
                    if parsed.fragment and target.suffix == ".md":
                        content = target.read_text()
                        headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.M)
                        anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                        self.assertIn(unquote(parsed.fragment), anchors)

    def test_homepage_story_order(self):
        text = (ROOT / "README.md").read_text()
        intro = text.index("## Introduction")
        eq = text.index(r"\max_w", intro)
        solver = text.index("iterative solver", eq)
        figure1 = text.index('src="assets/figure1.png"', solver)
        results = text.index("## Results", figure1)
        figure2 = text.index('src="assets/figure2.png"', results)
        self.assertLess(figure2, text.index("## Quick Start"))
        self.assertNotIn("reward-guided token weighting", text.lower())

    def test_badges_are_valid_local_svg(self):
        for asset in (ROOT / "assets").glob("badge-*.svg"):
            svg = ET.parse(asset).getroot()
            self.assertEqual(svg.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(svg.get("height"), "30")
            self.assertTrue(svg.get("aria-label"))

    def test_equation_delimiters_survive_markdown_escaping(self):
        for name in ("README.md", "README_zh.md"):
            with self.subTest(document=name):
                equation = (ROOT / name).read_text().split("$$")[1]
                # GFM can consume backslashes before punctuation such as braces.
                # Named delimiter commands survive that Markdown processing.
                markdown_processed = re.sub(r"\\([{}])", r"\1", equation)
                self.assertIn(r"\left\lbrace", markdown_processed)
                self.assertIn(r"\right\rbrace", markdown_processed)
                self.assertNotIn(r"\left{", markdown_processed)
                self.assertNotIn(r"\right}", markdown_processed)

    def test_framework_matches_source_manifest(self):
        lines = (ROOT / "docs/source-manifest.sha256").read_text().splitlines()
        self.assertGreater(len(lines), 100)
        for line in lines:
            digest, name = line.split("  ", 1)
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_bilingual_homepages_share_equation_figures_and_author_links(self):
        english = (ROOT / "README.md").read_text()
        chinese = (ROOT / "README_zh.md").read_text()
        self.assertEqual(english.split("$$")[1].strip(), chinese.split("$$")[1].strip())
        author_links = {
            "Zhenyu Wang": "https://zywang0701.github.io/",
            "Linjun Zhang": "https://linjunz.github.io/",
            "Yifan Hu": "https://sites.google.com/view/yifan-hu",
        }
        for text in (english, chinese):
            for name, url in author_links.items():
                self.assertIn(f"[{name}]({url})", text)
            for number in (1, 2):
                self.assertIn(f'src="assets/figure{number}.png"', text)

    def test_results_highlight_both_distillation_settings(self):
        for name, heading in (("README.md", "## Results"), ("README_zh.md", "## 实验结果")):
            with self.subTest(document=name):
                section = (ROOT / name).read_text().split(heading, 1)[1].split("\n## ", 1)[0]
                bullets = re.findall(r"^- .+$", section, re.M)
                self.assertEqual(len(bullets), 2)
                self.assertIn("strong-to-weak", bullets[0])
                self.assertIn("same-size", bullets[1])

    def test_public_docs_exclude_release_preparation_notes(self):
        documents = [ROOT / "README.md", ROOT / "README_zh.md", ROOT / "UPSTREAM.md",
                     ROOT / "LICENSE.md", *sorted((ROOT / "docs").glob("*.md"))]
        forbidden = ("preparation snapshot", "before announcing a public release",
                     "release-checklist.md", "user decision", "protocol_e030",
                     "token-interpretability setup", "author-provided", "not for public")
        for document in documents:
            with self.subTest(document=document.name):
                text = document.read_text().lower()
                for phrase in forbidden:
                    self.assertNotIn(phrase, text)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dr-opd-test-")
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name).resolve()
        self.env = os.environ.copy()
        for key in ("TRAIN_DATASET", "TEST_DATASET", "STEPS", "NGPU", "EVAL_STEPS", "STUDENT", "TEACHER",
                    "RESULTS_DIR", "FINAL_CKPT_DIR", "VENV", "WORK", "CUDA_VISIBLE_DEVICES"):
            self.env.pop(key, None)
        self.env.update(WORK=str(self.scratch / "work"), RESULTS_DIR=str(self.scratch / "results"))

    def run_launcher(self, *args, root=ROOT):
        return subprocess.run(["bash", str(root / "run.sh"), *args], cwd=self.scratch,
                              env=self.env, text=True, capture_output=True, timeout=15)

    def test_all_methods_have_side_effect_free_dry_run(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                result = self.run_launcher("train", arm, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Method: {arm}", result.stdout)
                self.assertIn("test_freq=1000", result.stdout)
                self.assertEqual(list(self.scratch.iterdir()), [])

    def test_dry_run_accepts_overrides_and_preserves_allocation(self):
        self.env.update(STEPS="30", STUDENT="/models/student with spaces", CUDA_VISIBLE_DEVICES="2,3")
        result = self.run_launcher("--dry-run", "train", "ours", "0.5", "7")
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ("Steps: 30", "Data seed: 7", "Lambda argument (ours): 0.5",
                         "CUDA_VISIBLE_DEVICES: 2,3", "Student: /models/student with spaces"):
            self.assertIn(expected, result.stdout)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_help_setup_and_data_dry_runs(self):
        for args in (("--help",), ("setup", "--dry-run"), ("data", "--dry-run")):
            result = self.run_launcher(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_invalid_input_fails_without_side_effects(self):
        cases = (("train", "../ours", "--dry-run"), ("train", "ours", "oops", "--dry-run"),
                 ("train", "ours", "0.4", "-2", "--dry-run"), ("train", "ours", "--unknown"))
        for args in cases:
            self.assertNotEqual(self.run_launcher(*args).returncode, 0)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_missing_data_fails_before_creating_runtime(self):
        result = self.run_launcher("train", "ours")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing prepared data", result.stderr)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def prepare_mock_training(self):
        root = self.scratch / "repo"
        (root / "OPDVR").mkdir(parents=True)
        shutil.copy2(ROOT / "run.sh", root / "run.sh")
        for source in (ROOT / "OPDVR").glob("*.sh"):
            shutil.copy2(source, root / "OPDVR" / source.name)
        bindir = self.scratch / "venv/bin"
        bindir.mkdir(parents=True)
        stub = ('#!/usr/bin/env bash\n'
                'printf "GPU_ALLOCATION=%s\\n" "${CUDA_VISIBLE_DEVICES:-}"\n'
                'printf "CHECKPOINTS=%s\\n" "${SAVE_STEPS:-}"\n'
                'printf "MOCK_TRAIN_ARG=%s\\n" "$@"\n'
                'exit "${MOCK_EXIT:-0}"\n')
        for name in ("python", "python3"):
            executable = bindir / name
            executable.write_text(stub)
            executable.chmod(0o755)
        # A Ray CLI invocation would fail the test; only the Python stub is allowed.
        ray = bindir / "ray"
        ray.write_text('#!/usr/bin/env bash\necho "Unexpected Ray CLI invocation" >&2\nexit 93\n')
        ray.chmod(0o755)
        data = self.scratch / "data with spaces"
        data.mkdir()
        (data / "train.parquet").touch()
        (data / "val.parquet").touch()
        self.env.update(VENV=str(bindir.parent), TRAIN_DATASET="data with spaces/train.parquet",
                        TEST_DATASET="data with spaces/val.parquet", CUDA_VISIBLE_DEVICES="4,5")
        return root

    def test_all_methods_reach_stub_without_conda_or_ray_cli(self):
        root = self.prepare_mock_training()
        for arm in ARMS:
            with self.subTest(arm=arm):
                result = self.run_launcher("train", arm, root=root)
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
                self.assertIn("GPU_ALLOCATION=4,5", result.stdout)
                self.assertIn("CHECKPOINTS=30,40,50", result.stdout)
                self.assertIn("MOCK_TRAIN_ARG=verl.trainer.main_ppo", result.stdout)
                self.assertIn(f"data.train_files={self.scratch}/data with spaces/train.parquet", result.stdout)

    def test_failure_is_not_hidden_by_tee(self):
        root = self.prepare_mock_training()
        self.env["MOCK_EXIT"] = "17"
        result = self.run_launcher("train", "ours", root=root)
        self.assertEqual(result.returncode, 17)

    def test_existing_log_is_not_overwritten(self):
        root = self.prepare_mock_training()
        self.assertEqual(self.run_launcher("train", "ours", root=root).returncode, 0)
        logfile = self.scratch / "results/math_ours_lam0.4_seed2.log"
        original = logfile.read_bytes()
        result = self.run_launcher("train", "ours", root=root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertEqual(logfile.read_bytes(), original)

    def test_shell_syntax(self):
        for source in [ROOT / "run.sh", *(ROOT / "scripts").glob("*.sh"), *(ROOT / "OPDVR").glob("*.sh")]:
            result = subprocess.run(["bash", "-n", str(source)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, f"{source}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
