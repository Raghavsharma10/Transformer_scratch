def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Visit the ``slice`` and a ``value`` and get the element."""
        value = self.visit(node=node.value)
        a_slice = self.visit(node=node.slice)

        result = value[a_slice]

        self.recomputed_values[node] = result
        return result