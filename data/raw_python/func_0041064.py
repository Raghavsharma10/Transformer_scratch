def matches(self, other):
        """Checks if another object is equivalent to this object.

        :returns: checks if another object is equivalent to this object
        :rtype: boolean
        """
        if not isinstance(other, Failure):
            return False
        if self.exc_info is None or other.exc_info is None:
            return self._matches(other)
        else:
            return self == other