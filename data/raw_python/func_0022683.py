def _projection(self, a, b, c):
        """Return projection of (a,b) onto (a,c)
        Arguments are point locations, not indexes.
        """
        ab = b - a
        ac = c - a
        return a + ((ab*ac).sum() / (ac*ac).sum()) * ac