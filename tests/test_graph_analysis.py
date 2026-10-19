from pathlib import Path

from ddya_detector.graph_analysis import (
    detect_plagiarism_communities,
    export_graph_to_html,
)


def test_detect_plagiarism_communities() -> None:
    """Verifies that the graph correctly groups students above the threshold."""
    comparisons = [
        ("alice.py", "bob.py", 0.95),  # Plagiarism
        ("alice.py", "charlie.py", 0.85),  # Plagiarism (Triangulation)
        ("bob.py", "charlie.py", 0.90),  # Plagiarism
        ("dave.py", "eve.py", 0.20),  # Innocent
        ("alice.py", "dave.py", 0.10),  # Innocent
        ("frank.py", "grace.py", 0.82),  # Plagiarism (Different group)
    ]

    communities = detect_plagiarism_communities(comparisons, threshold=0.8)

    assert len(communities) == 2

    assert {"alice.py", "bob.py", "charlie.py"} in communities
    assert {"frank.py", "grace.py"} in communities

    for community in communities:
        assert "dave.py" not in community
        assert "eve.py" not in community


def test_export_graph_to_html(tmp_path: Path) -> None:
    """Verifies that the HTML visualization is generated properly."""
    comparisons = [("alice.py", "bob.py", 0.95)]
    output_file = tmp_path / "plagiarism_network.html"

    export_graph_to_html(comparisons, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").lower()
    assert "<html" in content
    assert "alice.py" in content
    assert "bob.py" in content
