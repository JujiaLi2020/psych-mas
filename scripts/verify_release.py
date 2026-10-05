"""Check release versions, Demo integrity, and wheel runtime resources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile


def constant(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {name}: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert constant(root / "psymas_cli.py", "VERSION") == version
    assert constant(root / "psymas_ui/app_config.py", "APP_VERSION") == version
    assert f'"{version}"' in (root / "installer/PsyMAS.iss").read_text()
    assert f':-{version}' in (root / "docker-compose.release.yml").read_text()
    manifest = json.loads((root / "reproducibility/demo_manifest.json").read_text())
    assert manifest["package_software_version"] == version
    snapshot = root / manifest["snapshot"]["path"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest().lower() == manifest["snapshot"]["sha256"].lower()
    with zipfile.ZipFile(snapshot) as archive:
        assert archive.testzip() is None
        assert any(n.endswith(".sqlite") for n in archive.namelist())
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as wheel:
            names = wheel.namelist()
            for suffix in ["psymas_cli.py", "ui.py", "graph.py", "backend_service.py", "psymas_ui/case_prompts.py", "psymas_ui/report_validation.py", "psymas_ui/prompt_versions/v20.json", "config/rulebook_index.csv", "docker-compose.release.yml", snapshot.name]:
                assert any(n == suffix or n.endswith('/' + suffix) for n in names), suffix
            assert not any(n.startswith(("validation/", "tests/", "tools/", "release-build-")) for n in names)
            metadata = wheel.read(f"psych_mas-{version}.dist-info/METADATA").decode()
            assert f"Version: {version}" in metadata.splitlines()
    print(f"PsyMAS {version}: release versions, Demo integrity, and resources verified")


if __name__ == "__main__":
    main()
