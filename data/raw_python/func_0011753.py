def _tosubs(self, ixlist):
        """Maps a list of integer indices to sub-indices.
        ixlist can contain repeated indices and does not need to be sorted.
        Returns pair (ss, ms) where ss is a list of subsim numbers and ms is a
        list of lists of subindices m (one list for each subsim in ss).
        """
        n = len(ixlist)
        N = self._n
        ss = []
        ms = []
        if n == 0:
            return ss, ms
        j = 0 # the position in ixlist currently being processed
        ix = ixlist[j]
        if ix >= N or ix < -N:
            raise IndexError(
                    'index %d out of bounds for list of %d sims' % (ix, N))
        if ix < 0:
            ix += N
        while j < n:
            for s in range(0, self._n):
                low = self._si[s]
                high = self._si[s + 1]
                if ix >= low and ix < high:
                    ss.append(s)
                    msj = [ix - low]
                    j += 1
                    while j < n:
                        ix = ixlist[j]
                        if ix >= N or ix < -N:
                            raise IndexError(
                              'index %d out of bounds for list of %d sims' % (
                                ix, N))
                        if ix < 0:
                            ix += N
                        if ix < low or ix >= high:
                            break
                        msj.append(ix - low)
                        j += 1
                    ms.append(msj)
                if ix < low:
                    break
        return ss, ms