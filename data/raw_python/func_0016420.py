def deep_compare(self, other, settings):
        """
        Compares each field of the name one at a time to see if they match.
        Each name field has context-specific comparison logic.

        :param Name other: other Name for comparison
        :return bool: whether the two names are compatible
        """

        if not self._is_compatible_with(other):
            return False

        first, middle, last = self._compare_components(other, settings)

        return first and middle and last