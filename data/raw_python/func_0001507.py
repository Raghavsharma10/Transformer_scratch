def visit_NameConstant(self, node: ast.NameConstant) -> Any:
        """Forward the node value as a result."""
        self.recomputed_values[node] = node.value
        return node.value