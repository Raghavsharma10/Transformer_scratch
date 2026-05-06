def finditem(self, item, threshold=None):
        """Return most similar item to the provided one, or None if
        nothing exceeds the threshold.

        >>> from ngram import NGram
        >>> n = NGram([(0, "Spam"), (1, "Ham"), (2, "Eggsy"), (3, "Egggsy")],
        ...     key=lambda x:x[1].lower())
        >>> n.finditem((3, 'Hom'))
        (1, 'Ham')
        >>> n.finditem((4, "Oggsy"))
        (2, 'Eggsy')
        >>> n.finditem((4, "Oggsy"), 0.8)
        """
        results = self.searchitem(item, threshold)
        if results:
            return results[0][0]
        else:
            return None