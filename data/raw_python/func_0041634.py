def prev_listing(self, limit=None):
        """GETs previous :class:`Listing` directed to by this :class:`Listing`.  Returns :class:`Listing` object.
        
        :param limit: max number of entries to get
        """
        if self.before:
            return self._reddit._limit_get(self._path, eparams={'before': self.before}, limit=limit or self._limit)
        else:
            raise NoMoreError('no previous items')