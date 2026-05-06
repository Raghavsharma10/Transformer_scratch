def search(self, query, threshold=None):
        """Search the index for items whose key exceeds threshold
        similarity to the query string.

        :param query: returned items will have at least `threshold` \
        similarity to the query string.

        :return: list of pairs of (item, similarity) by decreasing similarity.

        >>> from ngram import NGram
        >>> n = NGram([(0, "SPAM"), (1, "SPAN"), (2, "EG")], key=lambda x:x[1])
        >>> sorted(n.search("SPA"))
        [((0, 'SPAM'), 0.375), ((1, 'SPAN'), 0.375)]
        >>> n.search("M")
        [((0, 'SPAM'), 0.125)]
        >>> n.search("EG")
        [((2, 'EG'), 1.0)]
        """
        threshold = threshold if threshold is not None else self.threshold
        results = []
        # Identify possible results
        for match, samegrams in self.items_sharing_ngrams(query).items():
            allgrams = (len(self.pad(query))
                        + self.length[match] - (2 * self.N) - samegrams + 2)
            similarity = self.ngram_similarity(samegrams, allgrams, self.warp)
            if similarity >= threshold:
                results.append((match, similarity))
        # Sort results by decreasing similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results