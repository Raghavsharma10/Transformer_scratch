def startswith(self, other):
        # type: (UrlPath) -> bool
        """
        Return True if this path starts with the other path.
        """
        try:
            other = UrlPath.from_object(other)
        except ValueError:
            raise TypeError('startswith first arg must be UrlPath, str, PathParam, not {}'.format(type(other)))
        else:
            return self._nodes[:len(other._nodes)] == other._nodes