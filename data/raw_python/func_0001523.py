def visit_ListComp(self, node: ast.ListComp) -> Any:
        """Compile the list comprehension as a function and call it."""
        result = self._execute_comprehension(node=node)

        for generator in node.generators:
            self.visit(generator.iter)

        self.recomputed_values[node] = result
        return result