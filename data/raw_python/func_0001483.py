def visit_Call(self, node: ast.Call) -> None:
        """Represent the call by dumping its source code."""
        if node in self._recomputed_values:
            value = self._recomputed_values[node]
            text = self._atok.get_text(node)

            self.reprs[text] = value

        self.generic_visit(node=node)