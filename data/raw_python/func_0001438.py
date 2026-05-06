def minor(self, minor: int) -> None:
        """
        param minor

        Minor version number property. Must be a non-negative integer.
        """
        self.filter_negatives(minor)
        self._minor = minor