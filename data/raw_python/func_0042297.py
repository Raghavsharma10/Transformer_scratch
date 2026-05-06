def visit_Identifier(self, node):
        """Mangle names."""

        if not self._is_mangle_candidate(node):
            return

        name = node.value
        symbol = node.scope.resolve(node.value)
        if symbol is None:
            self.free_variables.add(name)