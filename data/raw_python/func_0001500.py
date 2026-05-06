def visit_Num(self, node: ast.Num) -> Union[int, float]:
        """Recompute the value as the number at the node."""
        result = node.n

        self.recomputed_values[node] = result
        return result