from pathlib import Path

import networkx as nx
from pyvis.network import Network


def detect_plagiarism_communities(
    comparisons: list[tuple[str, str, float]], threshold: float = 0.8
) -> list[set[str]]:
    graph = nx.Graph()

    for student_a, student_b, score in comparisons:
        if score >= threshold:
            graph.add_edge(student_a, student_b, weight=score)
        else:
            graph.add_node(student_a)
            graph.add_node(student_b)

    communities: list[set[str]] = []

    for component in nx.connected_components(graph):
        if len(component) > 1:
            communities.append(set(component))

    return communities


def export_graph_to_html(
    comparisons: list[tuple[str, str, float]], output_path: Path, threshold: float = 0.8
) -> None:
    graph = nx.Graph()

    for student_a, student_b, score in comparisons:
        if score >= threshold:
            graph.add_edge(
                student_a,
                student_b,
                value=score,
                color="red",
                title=f"Similarity: {score:.1%}",
            )
        else:
            graph.add_node(student_a)
            graph.add_node(student_b)

    net = Network(notebook=False, directed=False, width="100%", height="750px")
    net.from_nx(graph)

    net.save_graph(str(output_path))
