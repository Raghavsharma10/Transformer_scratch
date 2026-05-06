def visit_List(self, node: ast.List) -> List[Any]:
        """Visit the elements and assemble the results into a list."""
        if isinstance(node.ctx, ast.Store):
            raise NotImplementedError("Can not compute the value of a Store on a list")

        result = [self.visit(node=elt) for elt in node.elts]

        self.recomputed_values[node] = result
        return result