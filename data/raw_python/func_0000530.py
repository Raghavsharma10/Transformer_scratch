def has_common(self, other):
        """Return set of common words between two word sets."""
        if not isinstance(other, WordSet):
            raise ValueError('Can compare only WordSets')
        return self.term_set & other.term_set