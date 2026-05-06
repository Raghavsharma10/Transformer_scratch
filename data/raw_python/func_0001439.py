def patch(self, patch: int) -> None:
        """
        param patch

        Patch version number property. Must be a non-negative integer.
        """
        self.filter_negatives(patch)
        self._patch = patch