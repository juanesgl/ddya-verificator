from pathlib import Path

import pytest
from ddya_detector.extractor import extract_features


def test_extract_features_valid_file(tmp_path: Path) -> None:
    # 1. Arrange: Prepare a simulated valid code file
    valid_code = '''"""This is the official homework docstring."""

def solve(x):
    counter = 0
    for i in range(x):
        if i % 2 == 0:
            counter += 1
    return counter

def main():
    solve(10)
'''

    file_path = tmp_path / "student_1.py"
    file_path.write_text(valid_code, encoding="utf-8")

    # 2. Act: Execute the extraction
    features = extract_features(file_path)

    # 3. Assert: Verify results
    assert features.docstring == "This is the official homework docstring."

    structure = features.solve_structures

    assert "FunctionDef" in structure
    assert "For" in structure
    assert "If" in structure
    assert "Return" in structure
    assert "counter" not in structure
    assert "x" not in structure


def test_extract_features_missing_functions(tmp_path: Path) -> None:
    """Verifies it rejects files that don't meet arena rules."""

    invalid_code = """def another_function():\n    pass"""
    file_path = tmp_path / "cheater.py"
    file_path.write_text(invalid_code, encoding="utf-8")

    with pytest.raises(ValueError, match="must contain both main and solve functions"):
        extract_features(file_path)


def test_extract_features_syntax_error(tmp_path: Path) -> None:
    """Verifies the pipeline doesn't crash if a student submits broken code."""

    broken_code = """def solve(x) -> \n    for in in range(x):\n"""
    file_path = tmp_path / "broken.py"
    file_path.write_text(broken_code, encoding="utf-8")

    with pytest.raises(ValueError, match="Syntax error"):
        extract_features(file_path)
