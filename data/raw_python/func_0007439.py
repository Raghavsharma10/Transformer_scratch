def union(self, *others):
        """Return the union of two or more sets as a new set.

        >>> from ngram import NGram
        >>> a = NGram(['spam', 'eggs'])
        >>> b = NGram(['spam', 'ham'])
        >>> sorted(list(a.union(b)))
        ['eggs', 'ham', 'spam']
        """
        return self.copy(super(NGram, self).union(*others))