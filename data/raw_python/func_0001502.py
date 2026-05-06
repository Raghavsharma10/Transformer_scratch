def visit_Bytes(self, node: ast.Bytes) -> bytes:
        """Recompute the value as the bytes at the node."""
        result = node.s

        self.recomputed_values[node] = result
        return node.s