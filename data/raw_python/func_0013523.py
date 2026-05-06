def getContinuousSetByName(self, name):
        """
        Returns the ContinuousSet with the specified name, or raises
        an exception otherwise.
        """
        if name not in self._continuousSetNameMap:
            raise exceptions.ContinuousSetNameNotFoundException(name)
        return self._continuousSetNameMap[name]