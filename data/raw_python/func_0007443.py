def symmetric_difference(self, other):
        """Return the symmetric difference of two sets as a new set.

        >>> from ngram import NGram
        >>> a = NGram(['spam', 'eggs'])
        >>> b = NGram(['spam', 'ham'])
        >>> sorted(list(a.symmetric_difference(b)))
        ['eggs', 'ham']
        """
        return self.copy(super(NGram, self).symmetric_difference(other))