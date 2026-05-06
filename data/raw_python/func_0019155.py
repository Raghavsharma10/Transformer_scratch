def shape(self) -> Tuple[int, ...]:
        """The shape of array |anntools.SeasonalANN.ratios|."""
        return tuple(int(sub) for sub in self.ratios.shape)