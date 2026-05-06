def Similarity(self, value=None):  # pylint: disable=C0103
        """Constructor for new default Similarities."""
        if value is None:
            value = 0.0
        return Similarity(value, threshold=self.threshold)