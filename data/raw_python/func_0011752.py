def _tosub(self, ix):
        """Given an integer index ix into the list of sims, returns the pair
        (s, m) where s is the relevant subsim and m is the subindex into s.
        So self[ix] == self._subsims[s][m]
        """
        N = self._n
        if ix >= N or ix < -N:
            raise IndexError(
                    'index %d out of bounds for list of %d sims' % (ix, N))
        if ix < 0:
            ix += N
        for s in range(0, self._n):
            if self._si[s + 1] - 1 >= ix:
                break
        m = ix - self._si[s]
        return s, m