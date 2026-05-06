def intersection(self, *others):
        """Return the intersection of two or more sets as a new set.

        >>> from ngram import NGram
        >>> a = NGram(['spam', 'eggs'])
        >>> b = NGram(['spam', 'ham'])
        >>> list(a.intersection(b))
        ['spam']
        """
        return self.copy(super(NGram, self).intersection(*others))