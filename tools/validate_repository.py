#!/usr/bin/env python3
"""Validate repository structure, naming, content, and checksum integrity."""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "file_manifest.tsv"
MUTABLE_RELEASE_METADATA = {
    ROOT / "README.md",
    ROOT / "CITATION.cff",
    ROOT / "docs" / "changelog.md",
}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
TEXT_SUFFIXES = {".cff", ".json", ".m", ".md", ".py", ".txt", ".tsv", ".yaml", ".yml"}
STANDARD_FILENAMES = {
    ".gitignore", "AUTHORS.md", "CITATION.cff", "CONTRIBUTING.md",
    "LICENSE", "Makefile", "README.md",
}
ACTION_PREFIXES = {
    "audit", "build", "compare", "diagnose", "estimate", "extract", "get",
    "prepare", "preflight", "run", "set", "smooth", "validate", "verify",
}
SNAKE_CASE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
PY_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
PASCAL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")

PROHIBITED_PUBLIC_TERMS = ["manu" + "script", "art" + "icle", "sub" + "mission"]
NONPORTABLE_MARKERS = ["/ho" + "me/", "/me" + "dia/", "/Us" + "ers/", "EXTERNAL" + "_USB"]
AUTOMATION_MARKERS = ["Chat" + "GPT", "Open" + "AI", "Clau" + "de", "Gem" + "ini", "Co" + "pilot"]
WORKING_PHRASES = ["recommended " + "next step", "next " + "command", "send " + "the output", "share " + "the output"]
INFORMAL_MARKERS = [
    r"\bTODO\b", r"\bFIXME\b", r"\bHACK\b", r"\bWIP\b", r"\bTBD\b",
    r"\boops\b", r"\bweird\b", r"\bmessy\b", r"\bhopefully\b",
    r"\bquick[- ]and[- ]dirty\b", r"\bfor now\b", r"\bjust in case\b",
]
STATUS_TERMS = ["fi" + "nal", "revi" + "sion", "ver" + "sion", r"v\d+(?:[._-]\d+)*"]
STATUS_NAME_PATTERN = re.compile(r"(^|[_-])(" + "|".join(STATUS_TERMS) + r")([_-]|\.|$)", re.IGNORECASE)

errors: list[str] = []


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def repository_files() -> list[Path]:
    output: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts):
            output.append(path)
    return sorted(output)


def logical_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".tsv.gz"):
        return name[:-7]
    return path.stem


def argparse_destinations(tree: ast.AST) -> tuple[set[str], set[str]]:
    defined: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            labels = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            destination = None
            for keyword in node.keywords:
                if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    destination = keyword.value.value
            if destination is None and labels:
                options = [label for label in labels if label.startswith("-")]
                destination = (max(options, key=len).lstrip("-") if options else labels[0]).replace("-", "_")
            if destination:
                defined.add(destination)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
            used.add(node.attr)
    return defined, used


files = repository_files()

# Path, transient-file, and naming checks.
for path in ROOT.rglob("*"):
    if path.is_dir() and path.name == "__pycache__":
        errors.append(f"Transient Python cache directory: {relative(path)}")
for path in files:
    rel = relative(path)
    if path.suffix.lower() in {".pyc", ".pyo"}:
        errors.append(f"Compiled Python artifact: {rel}")
    if path.stat().st_size == 0:
        errors.append(f"Empty file: {rel}")
    if STATUS_NAME_PATTERN.search(path.name):
        errors.append(f"Project-status label in filename: {rel}")

    if path.name not in STANDARD_FILENAMES and not (path.name == "README.md"):
        stem = logical_stem(path)
        if not SNAKE_CASE.fullmatch(stem):
            errors.append(f"Filename is not lowercase snake_case: {rel}")
    for directory in path.relative_to(ROOT).parts[:-1]:
        if directory == ".github":
            continue
        if not SNAKE_CASE.fullmatch(directory):
            errors.append(f"Directory is not lowercase snake_case: {rel} ({directory})")
            break

    if ("code" in path.parts or "tools" in path.parts) and path.suffix.lower() in {".py", ".m"}:
        prefix = logical_stem(path).split("_", 1)[0]
        if prefix not in ACTION_PREFIXES:
            errors.append(f"Code filename is not action-first: {rel}")

# Python syntax, identifiers, and argparse consistency.
python_paths = list((ROOT / "code").rglob("*.py")) + list((ROOT / "tools").rglob("*.py"))
for path in python_paths:
    rel = relative(path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception as error:
        errors.append(f"Python syntax error in {rel}: {error}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not PY_IDENTIFIER.fullmatch(node.name):
                errors.append(f"Non-snake-case Python function in {rel}:{node.lineno}: {node.name}")
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if argument.arg not in {"X", "M", "T"} and not PY_IDENTIFIER.fullmatch(argument.arg):
                    errors.append(f"Non-snake-case Python argument in {rel}:{node.lineno}: {argument.arg}")
        elif isinstance(node, ast.ClassDef) and not PASCAL_CASE.fullmatch(node.name):
            errors.append(f"Non-PascalCase Python class in {rel}:{node.lineno}: {node.name}")
    defined, used = argparse_destinations(tree)
    for name in sorted(used - defined):
        errors.append(f"Undefined argparse attribute in {rel}: args.{name}")

# MATLAB primary-function/file alignment.
for path in (ROOT / "code").rglob("*.m"):
    first_function = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("function "):
            declaration = stripped[len("function "):].strip()
            if "=" in declaration:
                declaration = declaration.split("=", 1)[1].strip()
            first_function = declaration.split("(", 1)[0].strip().split()[0]
            break
        if stripped and not stripped.startswith("%"):
            break
    if first_function and first_function != path.stem:
        errors.append(f"MATLAB primary function differs from filename: {relative(path)} -> {first_function}")

# Structured-file checks.
for path in files:
    rel = relative(path)
    if path.suffix.lower() == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"Invalid JSON in {rel}: {error}")
    if path.suffix.lower() == ".tsv" or path.name.endswith(".tsv.gz"):
        opener = gzip.open if path.name.endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8", newline="") as handle:
                widths = {len(row) for row in csv.reader(handle, delimiter="\t")}
            if not widths:
                errors.append(f"Empty TSV: {rel}")
            elif len(widths) != 1:
                errors.append(f"Nonrectangular TSV: {rel}")
        except Exception as error:
            errors.append(f"Unreadable TSV {rel}: {error}")

# Text-content checks, applied to every line of every public text file.
for path in files:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Makefile", ".gitignore"}:
        continue
    rel = relative(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    if "\r" in text:
        errors.append(f"Non-LF line ending in {rel}")
    lowered = text.lower()
    for term in PROHIBITED_PUBLIC_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            errors.append(f"Prohibited public term in {rel}: {term}")
    for marker in NONPORTABLE_MARKERS:
        if marker.lower() in lowered:
            errors.append(f"Nonportable path marker in {rel}")
    for marker in AUTOMATION_MARKERS:
        if marker.lower() in lowered:
            errors.append(f"Automation-provider marker in {rel}")
    for phrase in WORKING_PHRASES:
        if phrase.lower() in lowered:
            errors.append(f"Internal instruction phrase in {rel}")
    for pattern in INFORMAL_MARKERS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            line_number = text.count("\n", 0, match.start()) + 1
            errors.append(f"Informal working marker in {rel}:{line_number}: {match.group(0)}")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.rstrip(" \t") != line and path.suffix.lower() not in {".md", ".tsv"}:
            errors.append(f"Trailing whitespace in {rel}:{line_number}")

# Markdown links.
markdown_link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
for path in files:
    if path.suffix.lower() != ".md":
        continue
    for target in markdown_link.findall(path.read_text(encoding="utf-8")):
        target = target.strip().strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local_target = target.split("#", 1)[0]
        if not local_target:
            continue
        resolved = (path.parent / local_target).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"Markdown link escapes repository: {relative(path)} -> {target}")
            continue
        if not resolved.exists():
            errors.append(f"Broken Markdown link: {relative(path)} -> {target}")

# Source-data mapping.
source_map = ROOT / "results" / "reporting" / "source_data_map.tsv"
if source_map.exists():
    for row in csv.DictReader(source_map.open(encoding="utf-8"), delimiter="\t"):
        target = source_map.parent / row["source_file"]
        if not target.is_file():
            errors.append(f"Missing mapped source data: {row['source_file']}")
else:
    errors.append("Missing reporting source-data map.")

# Checksum manifest.
if not MANIFEST.exists():
    errors.append("Missing checksum manifest.")
else:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t"))
    expected = {row["path"]: (int(row["size_bytes"]), row["sha256"]) for row in rows}
    if len(expected) != len(rows):
        errors.append("Duplicate checksum-manifest path.")
    actual: dict[str, tuple[int, str]] = {}
    mutable_relative = {relative(path) for path in MUTABLE_RELEASE_METADATA}
    manifest_mutable = sorted(mutable_relative.intersection(expected))
    if manifest_mutable:
        errors.append(
            "Mutable release metadata must not be checksum-pinned: "
            + ", ".join(manifest_mutable)
        )
    for path in files:
        if path == MANIFEST or path in MUTABLE_RELEASE_METADATA:
            continue
        payload = path.read_bytes()
        actual[relative(path)] = (len(payload), hashlib.sha256(payload).hexdigest())
    for name, metadata in expected.items():
        if name not in actual:
            errors.append(f"Manifest path is missing: {name}")
        elif actual[name] != metadata:
            errors.append(f"Manifest checksum mismatch: {name}")
    for name in actual:
        if name not in expected:
            errors.append(f"File absent from checksum manifest: {name}")

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"Repository validation passed for {len(files)} files.")
