def _pickUpIteration(self, searchAnchor, objectsToSkip):
        """
        Picks up iteration from a previously provided page token. There are two
        different phases here:
        1) We are iterating over the initial set of intervals in which start
        is < the search start coorindate.
        2) We are iterating over the remaining intervals in which start >= to
        the search start coordinate.
        """
        self._searchAnchor = searchAnchor
        self._distanceFromAnchor = objectsToSkip
        self._searchIterator = self._search(
            searchAnchor,
            self._request.end if self._request.end != 0 else None)
        obj = next(self._searchIterator)
        if searchAnchor == self._request.start:
            # This is the initial set of intervals, we just skip forward
            # objectsToSkip positions
            for _ in range(objectsToSkip):
                obj = next(self._searchIterator)
        else:
            # Now, we are past this initial set of intervals.
            # First, we need to skip forward over the intervals where
            # start < searchAnchor, as we've seen these already.
            while self._getStart(obj) < searchAnchor:
                obj = next(self._searchIterator)
            # Now, we skip over objectsToSkip objects such that
            # start == searchAnchor
            for _ in range(objectsToSkip):
                if self._getStart(obj) != searchAnchor:
                    raise exceptions.BadPageTokenException
                obj = next(self._searchIterator)
        self._currentObject = obj
        self._nextObject = next(self._searchIterator, None)