def update(self, name):
        """
        Query a name for the relative index value to be added into the
        source map name field (optional 5th element).
        """

        if name is None:
            return

        if name not in self._names:
            # add the name if it isn't already tracked
            self._names[name] = len(self._names)

        result = self._names[name] - self._current
        self._current = self._names[name]
        return result