def _getDone(self, done, undone):
        """Parses the done|undone state.

        :done: Done marking pattern.
        :undone: Not done marking pattern.
        :returns: Pattern for done|undone or None if neither were specified.

        """
        if done:
            return self._getPattern(done, True)
        if undone:
            return self._getPattern(undone, False)