def _idForObject(self, defaultObject):
        """
        Generate an opaque identifier which can be used to talk about
        C{defaultObject}.

        @rtype: C{int}
        """
        identifier = self._allocateID()
        self._idsToObjects[identifier] = defaultObject
        return identifier