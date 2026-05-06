def getRnaQuantificationSetByName(self, name):
        """
        Returns the RnaQuantification set with the specified name, or raises
        an exception otherwise.
        """
        if name not in self._rnaQuantificationSetNameMap:
            raise exceptions.RnaQuantificationSetNameNotFoundException(name)
        return self._rnaQuantificationSetNameMap[name]