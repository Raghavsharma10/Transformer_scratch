def clear(self):
        """Remove all elements from this set.

        >>> from ngram import NGram
        >>> n = NGram(['spam', 'eggs'])
        >>> sorted(list(n))
        ['eggs', 'spam']
        >>> n.clear()
        >>> list(n)
        []
        """
        super(NGram, self).clear()
        self._grams = {}
        self.length = {}