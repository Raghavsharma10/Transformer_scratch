def items_sharing_ngrams(self, query):
        """Retrieve the subset of items that share n-grams the query string.

        :param query: look up items that share N-grams with this string.
        :return: mapping from matched string to the number of shared N-grams.

        >>> from ngram import NGram
        >>> n = NGram(["ham","spam","eggs"])
        >>> sorted(n.items_sharing_ngrams("mam").items())
        [('ham', 2), ('spam', 2)]
        """
        # From matched string to number of N-grams shared with query string
        shared = {}
        # Dictionary mapping n-gram to string to number of occurrences of that
        # ngram in the string that remain to be matched.
        remaining = {}
        for ngram in self.split(query):
            try:
                for match, count in self._grams[ngram].items():
                    remaining.setdefault(ngram, {}).setdefault(match, count)
                    # match as many occurrences as exist in matched string
                    if remaining[ngram][match] > 0:
                        remaining[ngram][match] -= 1
                        shared.setdefault(match, 0)
                        shared[match] += 1
            except KeyError:
                pass
        return shared