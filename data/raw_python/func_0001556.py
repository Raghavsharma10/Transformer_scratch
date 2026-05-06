def updaten(self, rangesets):
        """
        Update a rangeset with the union of itself and several others.
        """
        for rng in rangesets:
            if isinstance(rng, set):
                self.update(rng)
            else:
                self.update(RangeSet(rng))