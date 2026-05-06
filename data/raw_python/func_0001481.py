def visit_Name(self, node: ast.Name) -> None:
        """
        Resolve the name from the variable look-up and the built-ins.

        Due to possible branching (e.g., If-expressions), some nodes might lack the recomputed values. These nodes
        are ignored.
        """
        if node in self._recomputed_values:
            value = self._recomputed_values[node]

            # Check if it is a non-built-in
            is_builtin = True
            for lookup in self._variable_lookup:
                if node.id in lookup:
                    is_builtin = False
                    break

            if not is_builtin and _representable(value=value):
                text = self._atok.get_text(node)
                self.reprs[text] = value

        self.generic_visit(node=node)