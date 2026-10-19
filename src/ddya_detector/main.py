import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from ddya_detector.comparator import calculate_total_similarity
from ddya_detector.extractor import ExtractedFeatures, extract_features
from ddya_detector.graph_analysis import (
    detect_plagiarism_communities,
    export_graph_to_html,
)

# Set de parejas permitidas según el registro oficial
VALID_PAIRS = {
    frozenset({"alisson_gomez_c", "thaisg"}),
    frozenset({"david_ramirez_g", "jhoan_sierra_r"}),
    frozenset({"cristian_sua_c", "luisamolina_g"}),
    frozenset({"samuel_camargo_g", "tomas_corredor_v"}),
    frozenset({"juan_perez103", "juan_pinillos_b"}),
    frozenset({"santiago_marc_c", "juancanona"}),
    frozenset({"isabella_ardila", "joseromero"}),
    frozenset({"juan_vargas_d", "natha_bermudez_a"}),
    frozenset({"andrew_rodriguez", "juan_qcortes"}),
    frozenset({"juan_paez_z", "lesly_buitrago_l"}),
    frozenset({"nicolas_hachi", "juanpardo"}),
    frozenset({"david_barrero_r", "andres_guamba_h"}),
    frozenset({"samuel_hernandez", "iker_ardila_a1"}),
    frozenset({"aliss_gc23", "thaisg"}),
}

IGNORED_USERS = {"skinman95", "juanesgl"}


def extract_username(filename: str) -> str:
    """Extrae el usuario de VJUDGE del nombre del archivo."""
    parts = filename.split("_")
    return "_".join(parts[2:-1]).lower()


def main() -> None:
    """Orchestrates the plagiarism detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated Plagiarism Detector for Python Exercises"
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="submissions",
        help="Directory containing the submissions to analyze",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold between 0.0 and 1.0 (default: 0.8)",
    )
    args = parser.parse_args()

    submissions_dir = Path(args.directory)
    output_html = Path("plagiarism_network.html")
    threshold = args.threshold

    if not submissions_dir.exists():
        print(f"Error: Directory '{submissions_dir}' not found.")
        sys.exit(1)

    files = []
    for f in submissions_dir.rglob("*.py"):
        if "_AC_" in f.name:
            user = extract_username(f.stem)
            if user not in IGNORED_USERS:
                files.append(f)

    if len(files) < 2:
        print(f"Need at least 2 valid Python files in '{submissions_dir}' to compare.")
        sys.exit(1)

    print(f"Found {len(files)} accepted submissions. Extracting features...")

    features_map: dict[str, ExtractedFeatures] = {}
    file_to_exercise: dict[str, str] = {}
    structural_failures: list[tuple[str, str]] = []

    for file_path in tqdm(files, desc="Parsing ASTs"):
        try:
            features = extract_features(file_path)
            features_map[file_path.name] = features

            exercise_letter = file_path.stem.split("_")[-1]
            file_to_exercise[file_path.name] = f"Exercise {exercise_letter}"

        except ValueError as e:
            structural_failures.append((file_path.name, str(e)))

    exercise_groups = defaultdict(list)
    for file_name, exercise in file_to_exercise.items():
        if file_name in features_map:
            exercise_groups[exercise].append(file_name)

    print("\nCalculating similarities...")
    comparisons: list[tuple[str, str, float]] = []
    skipped_pairs = 0

    for exercise, students in exercise_groups.items():
        if len(students) < 2:
            continue

        pairs = list(combinations(students, 2))
        for student_a, student_b in tqdm(pairs, desc=f"Comparing {exercise}"):
            user_a = extract_username(Path(student_a).stem)
            user_b = extract_username(Path(student_b).stem)

            if frozenset({user_a, user_b}) in VALID_PAIRS:
                skipped_pairs += 1
                continue

            score = calculate_total_similarity(
                features_map[student_a], features_map[student_b]
            )
            comparisons.append((student_a, student_b, score))

    if skipped_pairs > 0:
        print(f"\nSkipped {skipped_pairs} valid laboratory partnerships.")

    print("\nBuilding network graph...")
    communities = detect_plagiarism_communities(comparisons, threshold)
    export_graph_to_html(comparisons, output_html, file_to_exercise, threshold)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if structural_failures:
        print("STRUCTURAL SPECIFICATION FAILURES:")
        for file_name, reason in structural_failures:
            print(f"  - {file_name}: Missing 'main' or 'solve'")
        print("-" * 50)

    print("PLAGIARISM DETECTED:")
    if not communities:
        print(f"  No plagiarism communities found (Threshold: {threshold:.0%})")
    else:
        print(f"  Detected {len(communities)} suspicious groups:\n")
        for i, group in enumerate(communities, 1):
            print(f"  Group {i}: {', '.join(group)}")

    print("=" * 50)
    print(f"Interactive graph saved to: {output_html.absolute()}")


if __name__ == "__main__":
    main()
