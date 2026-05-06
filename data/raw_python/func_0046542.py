def has_readonly(self, s):
        """Tests whether store `s` is read-only."""
        for t in self.transitions:
            if list(t.lhs[s]) != list(t.rhs[s]):
                return False
        return True