def visit_Slice(self, node: ast.Slice) -> slice:
        """Visit ``lower``, ``upper`` and ``step`` and recompute the node as a ``slice``."""
        lower = None  # type: Optional[int]
        if node.lower is not None:
            lower = self.visit(node=node.lower)

        upper = None  # type: Optional[int]
        if node.upper is not None:
            upper = self.visit(node=node.upper)

        step = None  # type: Optional[int]
        if node.step is not None:
            step = self.visit(node=node.step)

        result = slice(lower, upper, step)

        self.recomputed_values[node] = result
        return result