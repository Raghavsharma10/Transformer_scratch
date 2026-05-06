def visit_Tuple(self, node: ast.Tuple) -> Tuple[Any, ...]:
        """Visit the elements and assemble the results into a tuple."""
        if isinstance(node.ctx, ast.Store):
            raise NotImplementedError("Can not compute the value of a Store on a tuple")

        result = tuple(self.visit(node=elt) for elt in node.elts)

        self.recomputed_values[node] = result
        return result