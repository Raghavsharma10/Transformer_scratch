def update(self) -> None:
        """Update subclass of |RelSubweightsMixin| based on `refweights`."""
        mask = self.mask
        weights = self.refweights[mask]
        self[~mask] = numpy.nan
        self[mask] = weights/numpy.sum(weights)