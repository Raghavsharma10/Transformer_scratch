def copy(self, items=None):
        """Return a new NGram object with the same settings, and
        referencing the same items.  Copy is shallow in that
        each item is not recursively copied.   Optionally specify
        alternate items to populate the copy.

        >>> from ngram import NGram
        >>> from copy import deepcopy
        >>> n = NGram(['eggs', 'spam'])
        >>> m = n.copy()
        >>> m.add('ham')
        >>> sorted(list(n))
        ['eggs', 'spam']
        >>> sorted(list(m))
        ['eggs', 'ham', 'spam']
        >>> p = n.copy(['foo', 'bar'])
        >>> sorted(list(p))
        ['bar', 'foo']
        """
        return NGram(items if items is not None else self,
                     self.threshold, self.warp, self._key,
                     self.N, self._pad_len, self._pad_char)