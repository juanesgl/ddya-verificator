import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedFeatures:
    docstring: str
    solve_structures: list[str]


class StructureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.structure: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.structure.append(type(node).__name__)

        ast.NodeVisitor.generic_visit(self, node)


def extract_features(file_path: Path) -> ExtractedFeatures:
    try:
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in {file_path.name}: {e}")
    except Exception as e:
        raise ValueError(f"Cannot read {file_path.name}: {e}")

    docstring = ast.get_docstring(tree) or ""

    solve_node: ast.FunctionDef | None = None
    has_main = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "solve":
                solve_node = node
            elif node.name == "main":
                has_main = True

    if not solve_node or not has_main:
        raise ValueError(
            f"Incomplete file: {file_path.name} must contain "
            "both main and solve functions"
        )

    visitor = StructureVisitor()
    visitor.visit(solve_node)

    return ExtractedFeatures(
        docstring=docstring,
        solve_structures=visitor.structure,
    )
