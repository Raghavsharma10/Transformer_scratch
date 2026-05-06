def _initialiseIteration(self):
        """
        Starts a new iteration.
        """
        self._searchIterator = self._search(
            self._request.start,
            self._request.end if self._request.end != 0 else None)
        self._currentObject = next(self._searchIterator, None)
        if self._currentObject is not None:
            self._nextObject = next(self._searchIterator, None)
            self._searchAnchor = self._request.start
            self._distanceFromAnchor = 0
            firstObjectStart = self._getStart(self._currentObject)
            if firstObjectStart > self._request.start:
                self._searchAnchor = firstObjectStart