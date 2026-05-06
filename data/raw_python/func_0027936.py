def _sequenceContainer(self, store):
        """
        Smash whatever we got into a list and save the result in case we are
        executed multiple times.  This keeps us from tripping up over
        generators and the like.
        """
        if self._sequence is None:
            self._sequence = list(self.container)
            self._clause = ', '.join(['?'] * len(self._sequence))
        return self._clause