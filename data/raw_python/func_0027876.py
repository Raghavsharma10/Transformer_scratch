def get_pair(self, term1, term2):

        """
        Get the value for a pair of terms.

        Args:
            term1 (str)
            term2 (str)

        Returns:
            The stored value.
        """

        key = self.key(term1, term2)
        return self.pairs.get(key, None)