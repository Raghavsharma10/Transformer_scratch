def set_pair(self, term1, term2, value, **kwargs):

        """
        Set the value for a pair of terms.

        Args:
            term1 (str)
            term2 (str)
            value (mixed)
        """

        key = self.key(term1, term2)
        self.keys.update([term1, term2])
        self.pairs[key] = value