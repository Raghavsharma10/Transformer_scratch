def getReferenceByName(self, name):
        """
        Returns the reference with the specified name.
        """
        if name not in self._referenceNameMap:
            raise exceptions.ReferenceNameNotFoundException(name)
        return self._referenceNameMap[name]