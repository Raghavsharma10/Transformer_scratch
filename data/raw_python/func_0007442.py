def intersection_update(self, *others):
        """Update the set with the intersection of itself and other sets.

        >>> from ngram import NGram
        >>> n = NGram(['spam', 'eggs'])
        >>> other = set(['spam', 'ham'])
        >>> n.intersection_update(other)
        >>> list(n)
        ['spam']
        """
        self.difference_update(super(NGram, self).difference(*others))