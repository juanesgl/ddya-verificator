import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ddya_detector.extractor import ExtractedFeatures


def calculate_structure_similarity(struc_a: list[str], struct_b: list[str]) -> float:
    if not struc_a and not struct_b:
        return 1.0

    str_a = "-".join(struc_a)
    str_b = "-".join(struct_b)

    return float(Levenshtein.ratio(str_a, str_b))


def calculate_text_similarity(text_a: str, text_b: str) -> float:
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    vectorizer = TfidfVectorizer()

    try:
        tfdf_matrix = vectorizer.fit_transform([text_a, text_b])

        similarity = cosine_similarity(tfdf_matrix[0:1], tfdf_matrix[1:2])[0][0]
        return float(similarity)
    except ValueError:
        return 0.0


def calculate_total_similarity(
    features_a: ExtractedFeatures,
    features_b: ExtractedFeatures,
    structure_weight: float = 0.7,
    text_weight: float = 0.3,
) -> float:
    struct_sim = calculate_structure_similarity(
        features_a.solve_structures, features_b.solve_structures
    )

    text_sim = calculate_text_similarity(features_a.docstring, features_b.docstring)

    return (struct_sim * structure_weight) + (text_sim * text_weight)
