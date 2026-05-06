def visit_Attribute(self, node: ast.Attribute) -> None:
        """Represent the attribute by dumping its source code."""
        if node in self._recomputed_values:
            value = self._recomputed_values[node]

            if _representable(value=value):
                text = self._atok.get_text(node)
                self.reprs[text] = value

        self.generic_visit(node=node)