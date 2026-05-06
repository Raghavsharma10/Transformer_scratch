def visit_Set(self, node: ast.Set) -> Set[Any]:
        """Visit the elements and assemble the results into a set."""
        result = set(self.visit(node=elt) for elt in node.elts)

        self.recomputed_values[node] = result
        return result