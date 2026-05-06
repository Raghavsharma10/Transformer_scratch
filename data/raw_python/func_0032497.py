def _newIdentifier(self):
        """
        Make a new identifier for an as-yet uncreated model object.

        @rtype: C{int}
        """
        id = self._allocateID()
        self._idsToObjects[id] = self._NO_OBJECT_MARKER
        self._lastValues[id] = None
        return id