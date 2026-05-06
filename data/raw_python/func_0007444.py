def symmetric_difference_update(self, other):
        """Update the set with the symmetric difference of itself and `other`.

        >>> from ngram import NGram
        >>> n = NGram(['spam', 'eggs'])
        >>> other = set(['spam', 'ham'])
        >>> n.symmetric_difference_update(other)
        >>> sorted(list(n))
        ['eggs', 'ham']
        """
        intersection = super(NGram, self).intersection(other)
        self.update(other)  # add items present in other
        self.difference_update(intersection)