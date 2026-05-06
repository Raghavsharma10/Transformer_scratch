def major(self, major: int) -> None:
        """
        param major

        Major version number property. Must be a non-negative integer.
        """
        self.filter_negatives(major)
        self._major = major