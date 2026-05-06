def cloneURL(self, scheme, netloc, pathsegs, querysegs, fragment):
        """
        Override the base implementation to pass along the share ID our
        constructor was passed.
        """
        return self.__class__(
            self._shareID, scheme, netloc, pathsegs, querysegs, fragment)