def simplified(self):
        """A simplified representation of the same transformation.
        """
        if self._simplified is None:
            self._simplified = SimplifiedChainTransform(self)
        return self._simplified