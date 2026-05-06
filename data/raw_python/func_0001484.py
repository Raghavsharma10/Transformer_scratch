def visit_ListComp(self, node: ast.ListComp) -> None:
        """Represent the list comprehension by dumping its source code."""
        if node in self._recomputed_values:
            value = self._recomputed_values[node]
            text = self._atok.get_text(node)

            self.reprs[text] = value

        self.generic_visit(node=node)