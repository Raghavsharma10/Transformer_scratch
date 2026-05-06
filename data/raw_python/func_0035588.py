def rank(self):
        """convert a list of integers so that the lowest integer
        is 0, the next lowest is 1 ...
        note: modifies list in place"""
        # XXX FIX ME, should the lowest value be 1 or 0?
        symclasses = self.symclasses
        stableSort = map(None, symclasses, range(len(symclasses)))
        stableSort.sort()

        last = None
        x = -1
        for order, i in stableSort:
            if order != last:
                x += 1
                last = order
            symclasses[i] = x