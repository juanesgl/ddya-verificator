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
    comparisons: list[tuple[str, str, float]],
    output_path: Path,
    node_groups: dict[str, str],
    threshold: float = 0.8,
) -> None:
    graph = nx.Graph()

    for node, group in node_groups.items():
        graph.add_node(node, group=group, title=f"{group}\n{node}")

    for student_a, student_b, score in comparisons:
        if score >= threshold:
            graph.add_edge(
                student_a,
                student_b,
                weight=score,
                width=3,
                color="red",
                title=f"Plagiarism Alert: {score:.1%}",
            )
        else:
            graph.add_edge(
                student_a,
                student_b,
                weight=score,
                width=1,
                color="rgba(0, 0, 0, 0.80)",
                title=f"Similarity: {score:.1%}",
            )

    # Initialize PyVis network
    net = Network(notebook=False, directed=False, width="100%", height="100vh")
    net.from_nx(graph)

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=250,
        spring_strength=0.05,
        damping=0.4,
        overlap=0,
    )

    # net.show_buttons(filter_=['physics'])
    net.save_graph(str(output_path))
