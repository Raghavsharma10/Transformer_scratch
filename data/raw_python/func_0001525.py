def visit_DictComp(self, node: ast.DictComp) -> Any:
        """Compile the dictionary comprehension as a function and call it."""
        result = self._execute_comprehension(node=node)

        for generator in node.generators:
            self.visit(generator.iter)

        self.recomputed_values[node] = result
        return result