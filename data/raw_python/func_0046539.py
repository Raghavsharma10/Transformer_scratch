def match(self, other):
        """Returns true iff self (as a pattern) matches other (as a
        configuration). Note that this is asymmetric: other is allowed
        to have symbols that aren't found in self."""

        if len(self) != len(other):
            raise ValueError()
        for s1, s2 in zip(self, other):
            i = s2.position - s1.position
            if i < 0:
                return False
            n = len(s1)
            while i+n > len(s2) and s1[n-1] == syntax.BLANK:
                n -= 1
            if s2.values[i:i+n] != s1.values[:n]:
                return False
        return True