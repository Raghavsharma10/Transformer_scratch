def findLowest(self, symorders):
        """Find the position of the first lowest tie in a
        symorder or -1 if there are no ties"""
        _range = range(len(symorders))
        stableSymorders = map(None, symorders, _range)

        # XXX FIX ME
        # Do I need to sort?
        stableSymorders.sort()
        lowest = None
        for index in _range:
            if stableSymorders[index][0] == lowest:
                return stableSymorders[index-1][1]
            lowest = stableSymorders[index][0]
        return -1