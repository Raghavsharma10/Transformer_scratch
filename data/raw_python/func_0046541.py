def has_stack(self, s):
        """Tests whether store `s` is a stack, that is, it never moves from
        position 0."""
        for t in self.transitions:
            if t.lhs[s].position != 0:
                return False
            if t.rhs[s].position != 0:
                return False
        return True