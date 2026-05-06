def searchitem(self, item, threshold=None):
        """Search the index for items whose key exceeds the threshold
        similarity to the key of the given item.

        :return: list of pairs of (item, similarity) by decreasing similarity.

        >>> from ngram import NGram
        >>> n = NGram([(0, "SPAM"), (1, "SPAN"), (2, "EG"),
        ... (3, "SPANN")], key=lambda x:x[1])
        >>> sorted(n.searchitem((2, "SPA"), 0.35))
        [((0, 'SPAM'), 0.375), ((1, 'SPAN'), 0.375)]
        """
        return self.search(self.key(item), threshold)