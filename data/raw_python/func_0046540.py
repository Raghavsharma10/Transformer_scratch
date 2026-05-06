def has_cell(self, s):
        """Tests whether store `s` is a cell, that is, it uses exactly one
        cell, and there can take on only a finite number of states)."""

        for t in self.transitions:
            if len(t.lhs[s]) != 1:
                return False
            if len(t.rhs[s]) != 1:
                return False
            if t.lhs[s].position != 0:
                return False
            if t.rhs[s].position != 0:
                return False
        return True