def visit_Str(self, node: ast.Str) -> str:
        """Recompute the value as the string at the node."""
        result = node.s

        self.recomputed_values[node] = result
        return result