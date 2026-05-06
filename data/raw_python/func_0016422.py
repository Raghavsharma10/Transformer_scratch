def _is_compatible_with(self, other):
        """
        Return True if names are not incompatible.

        This checks that the gender of titles and compatibility of suffixes
        """

        title = self._compare_title(other)
        suffix = self._compare_suffix(other)

        return title and suffix