def getCallSetByName(self, name):
        """
        Returns a CallSet with the specified name, or raises a
        CallSetNameNotFoundException if it does not exist.
        """
        if name not in self._callSetNameMap:
            raise exceptions.CallSetNameNotFoundException(name)
        return self._callSetNameMap[name]