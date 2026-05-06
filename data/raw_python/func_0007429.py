def _split(self, string):
        """Iterates over the ngrams of a string (no padding).

        >>> from ngram import NGram
        >>> n = NGram()
        >>> list(n._split("hamegg"))
        ['ham', 'ame', 'meg', 'egg']
        """
        for i in range(len(string) - self.N + 1):
            yield string[i:i + self.N]