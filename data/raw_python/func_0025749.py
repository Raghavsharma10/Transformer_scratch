def getPosArgs(self):
        """ Return a list, in order, of any parameters marked with "pos=N" in
            the .cfgspc file. """
        if len(self._posArgs) < 1: return []
        # The first item in the tuple is the index, so we now sort by it
        self._posArgs.sort()
        # Build a return list
        retval = []
        for idx, scope, name in self._posArgs:
            theDict, val = findScopedPar(self, scope, name)
            retval.append(val)
        return retval