def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        """Compile the generator expression as a function and call it."""
        result = self._execute_comprehension(node=node)

        for generator in node.generators:
            self.visit(generator.iter)

        # Do not set the computed value of the node since its representation would be non-informative.
        return result