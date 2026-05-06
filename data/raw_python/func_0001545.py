def _folded_slices(self):
        """Internal generator that is able to retrieve ranges organized by step.
        Complexity: O(n) with n = number of ranges in tree."""
        if len(self) == 0:
            return

        prng = None         # pending range
        istart = None       # processing starting indice
        m = 0               # processing step
        for sli in self._contiguous_slices():
            start = sli.start
            stop = sli.stop
            unitary = (start + 1 == stop)   # one indice?
            if istart is None:  # first loop
                if unitary:
                    istart = start
                else:
                    prng = [start, stop, 1]
                    istart = stop - 1
                i = k = istart
            elif m == 0:        # istart is set but step is unknown
                if not unitary:
                    if prng is not None:
                        # yield and replace pending range
                        yield slice(*prng)
                    else:
                        yield slice(istart, istart + 1, 1)
                    prng = [start, stop, 1]
                    istart = k = stop - 1
                    continue
                i = start
            else:               # step m > 0
                assert m > 0
                i = start
                # does current range lead to broken step?
                if m != i - k or not unitary:
                    #j = i if m == i - k else k
                    if m == i - k: j = i
                    else: j = k
                    # stepped is True when autostep setting does apply
                    stepped = (j - istart >= self._autostep * m)
                    if prng:    # yield pending range?
                        if stepped:
                            prng[1] -= 1
                        else:
                            istart += m
                        yield slice(*prng)
                        prng = None
                if m != i - k:
                    # case: step value has changed
                    if stepped:
                        yield slice(istart, k + 1, m)
                    else:
                        for j in range(istart, k - m + 1, m):
                            yield slice(j, j + 1, 1)
                        if not unitary:
                            yield slice(k, k + 1, 1)
                    if unitary:
                        if stepped:
                            istart = i = k = start
                        else:
                            istart = k
                    else:
                        prng = [start, stop, 1]
                        istart = i = k = stop - 1
                elif not unitary:
                    # case: broken step by contiguous range
                    if stepped:
                        # yield 'range/m' by taking first indice of new range
                        yield slice(istart, i + 1, m)
                        i += 1
                    else:
                        # autostep setting does not apply in that case
                        for j in range(istart, i - m + 1, m):
                            yield slice(j, j + 1, 1)
                    if stop > i + 1:    # current->pending only if not unitary
                        prng = [i, stop, 1]
                    istart = i = k = stop - 1
            m = i - k   # compute step
            k = i
        # exited loop, process pending range or indice...
        if m == 0:
            if prng:
                yield slice(*prng)
            else:
                yield slice(istart, istart + 1, 1)
        else:
            assert m > 0
            stepped = (k - istart >= self._autostep * m)
            if prng:
                if stepped:
                    prng[1] -= 1
                else:
                    istart += m
                yield slice(*prng)
                prng = None
            if stepped:
                yield slice(istart, i + 1, m)
            else:
                for j in range(istart, i + 1, m):
                    yield slice(j, j + 1, 1)