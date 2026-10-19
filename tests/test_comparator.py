from ddya_detector.comparator import (
    calculate_structure_similarity,
    calculate_text_similarity,
    calculate_total_similarity,
)
from ddya_detector.extractor import ExtractedFeatures


def test_calculate_structure_similarity() -> None:
    """Verifies Levenshtein distance handles AST sequences correctly."""
    seq1 = ["FunctionDef", "For", "If", "Assign", "Return"]
    seq2 = ["FunctionDef", "For", "If", "Assign", "Return"]
    seq_diff = ["FunctionDef", "Return"]

    assert calculate_structure_similarity(seq1, seq2) == 1.0
    assert calculate_structure_similarity(seq1, seq_diff) < 1.0
    assert calculate_structure_similarity([], []) == 1.0


def test_calculate_text_similarity() -> None:
    """Verifies TF-IDF and Cosine Similarity for natural language."""
    text1 = "This is the official explanation of the algorithm."
    text2 = "This is the official explanation of the algorithm."
    text_diff = "Completely unrelated text about a different topic."

    assert calculate_text_similarity(text1, text2) > 0.99
    assert calculate_text_similarity(text1, text_diff) < 0.5
    assert calculate_text_similarity("", "") == 1.0
    assert calculate_text_similarity("Test text", "") == 0.0


def test_calculate_total_similarity() -> None:
    """Verifies the weighted average of both metrics."""
    feat_a = ExtractedFeatures(docstring="Hello world", solve_structures=["For", "If"])
    feat_b = ExtractedFeatures(docstring="Hello world", solve_structures=["For", "If"])

    score = calculate_total_similarity(
        feat_a, feat_b, structure_weight=0.7, text_weight=0.3
    )
    assert score == 1.0
