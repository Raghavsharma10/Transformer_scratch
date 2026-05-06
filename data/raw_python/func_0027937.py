def _sequenceArgs(self, store):
        """
        Filter each element of the data using the attribute type being
        tested for containment and hand back the resulting list.
        """
        self._sequenceContainer(store) # Force _sequence to be valid
        return [self.attribute.infilter(pyval, None, store) for pyval in self._sequence]