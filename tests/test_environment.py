"""Environment and Project Structure Verification Tests."""

import importlib
import sys
import unittest
from pathlib import Path


class TestProjectEnvironment(unittest.TestCase):
    """Verify environment setup, configuration, and directory layout."""

    def setUp(self):
        """Locate project root directory."""
        self.base_dir = Path(__file__).resolve().parent.parent

    def test_python_version(self):
        """Verify that Python version meets the minimum requirement (>= 3.10)."""
        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 10),
            f"Python version {sys.version} is below minimum requirement 3.10",
        )

    def test_directory_structure(self):
        """Verify that all core project directories exist."""
        expected_directories = [
            self.base_dir / "data" / "raw",
            self.base_dir / "data" / "processed",
            self.base_dir / "notebooks",
            self.base_dir / "sql",
            self.base_dir / "src",
            self.base_dir / "app",
            self.base_dir / "tests",
        ]

        for directory in expected_directories:
            with self.subTest(directory=str(directory.relative_to(self.base_dir))):
                self.assertTrue(directory.exists(), f"Missing required directory: {directory.relative_to(self.base_dir)}")
                self.assertTrue(directory.is_dir(), f"Path is not a directory: {directory.relative_to(self.base_dir)}")

    def test_configuration_files(self):
        """Verify that basic configuration and documentation files exist."""
        expected_files = [
            self.base_dir / "requirements.txt",
            self.base_dir / ".gitignore",
            self.base_dir / "README.md",
            self.base_dir / "pyproject.toml",
            self.base_dir / "docs" / "GIT_WORKFLOW.md",
            self.base_dir / ".github" / "PULL_REQUEST_TEMPLATE.md",
            self.base_dir / "docs" / "DATA_DICTIONARY.md",
        ]

        for file_path in expected_files:
            with self.subTest(file=str(file_path.relative_to(self.base_dir))):
                self.assertTrue(file_path.exists(), f"Missing required file: {file_path.relative_to(self.base_dir)}")
                self.assertTrue(file_path.is_file(), f"Path is not a file: {file_path.relative_to(self.base_dir)}")

    def test_package_structure(self):
        """Verify that src and app are valid importable Python packages."""
        if str(self.base_dir) not in sys.path:
            sys.path.insert(0, str(self.base_dir))

        src_pkg = importlib.import_module("src")
        app_pkg = importlib.import_module("app")

        self.assertIsNotNone(src_pkg)
        self.assertIsNotNone(app_pkg)


if __name__ == "__main__":
    unittest.main()
