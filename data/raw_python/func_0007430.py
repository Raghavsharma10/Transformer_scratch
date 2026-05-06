def add(self, item):
        """Add an item to the N-gram index (if it has not already been added).

        >>> from ngram import NGram
        >>> n = NGram()
        >>> n.add("ham")
        >>> list(n)
        ['ham']
        >>> n.add("spam")
        >>> sorted(list(n))
        ['ham', 'spam']
        """
        if item not in self:
            # Add the item to the base set
            super(NGram, self).add(item)
            # Record length of padded string
            padded_item = self.pad(self.key(item))
            self.length[item] = len(padded_item)
            for ngram in self._split(padded_item):
                # Add a new n-gram and string to index if necessary
                self._grams.setdefault(ngram, {}).setdefault(item, 0)
                # Increment number of times the n-gram appears in the string
                self._grams[ngram][item] += 1