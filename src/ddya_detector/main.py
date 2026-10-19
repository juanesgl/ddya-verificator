import sys
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from ddya_detector.comparator import calculate_total_similarity
from ddya_detector.extractor import ExtractedFeatures, extract_features
from ddya_detector.graph_analysis import (
    detect_plagiarism_communities,
    export_graph_to_html,
)


def main() -> None:
    submissions_dir = Path("submissions")
    output_html = Path("plagiarism_network.html")
    threshold = 0.8

    if not submissions_dir.exists():
        print(f"Error: Directory '{submissions_dir}' not found.")
        sys.exit(1)

    files = [f for f in submissions_dir.rglob("*.py") if "_AC_" in f.name]
    if len(files) < 2:
        print("Need at least 2 Python files in 'submissions/' to compare.")
        sys.exit(1)

    print(f"Found {len(files)} submissions. Extracting features...")

    features_map: dict[str, ExtractedFeatures] = {}
    for file_path in tqdm(files, desc="Parsing ASTs"):
        try:
            features = extract_features(file_path)
            features_map[file_path.name] = features
        except ValueError as e:
            print(f"\n[!] Skipping {file_path.name}: {e}")

    valid_files = list(features_map.keys())
    if len(valid_files) < 2:
        print("\nNot enough valid files remaining for comparison.")
        sys.exit(1)

    print("\nCalculating similarities...")

    comparisons: list[tuple[str, str, float]] = []
    pairs = list(combinations(valid_files, 2))

    for student_a, student_b in tqdm(pairs, desc="Comparing Pairs"):
        score = calculate_total_similarity(
            features_map[student_a], features_map[student_b]
        )
        comparisons.append((student_a, student_b, score))

    print("\nBuilding network graph...")

    communities = detect_plagiarism_communities(comparisons, threshold)
    export_graph_to_html(comparisons, output_html, threshold)

    print("\n" + "=" * 40)
    print("PLAGIARISM DETECTION RESULTS")
    print("=" * 40)
    if not communities:
        print(f"No plagiarism detected (Threshold: {threshold:.0%})")
    else:
        print(f"Detected {len(communities)} suspicious communities:\n")
        for i, group in enumerate(communities, 1):
            print(f"Group {i}: {', '.join(group)}")

    print("-" * 40)
    print(f"Interactive graph saved to: {output_html.absolute()}")


main()
