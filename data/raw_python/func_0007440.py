def difference(self, *others):
        """Return the difference of two or more sets as a new set.

        >>> from ngram import NGram
        >>> a = NGram(['spam', 'eggs'])
        >>> b = NGram(['spam', 'ham'])
        >>> list(a.difference(b))
        ['eggs']
        """
        return self.copy(super(NGram, self).difference(*others))