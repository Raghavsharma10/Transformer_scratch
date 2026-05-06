def visit_Index(self, node: ast.Index) -> Any:
        """Visit the node's ``value``."""
        result = self.visit(node=node.value)

        self.recomputed_values[node] = result
        return result