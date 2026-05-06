def prune(self, wordlist):
        """
        Prune the current reach instance by removing items.

        Parameters
        ----------
        wordlist : list of str
            A list of words to keep. Note that this wordlist need not include
            all words in the Reach instance. Any words which are in the
            wordlist, but not in the reach instance are ignored.

        """
        # Remove duplicates
        wordlist = set(wordlist).intersection(set(self.items.keys()))
        indices = [self.items[w] for w in wordlist if w in self.items]
        if self.unk_index is not None and self.unk_index not in indices:
            raise ValueError("Your unknown item is not in your list of items. "
                             "Set it to None before pruning, or pass your "
                             "unknown item.")
        self.vectors = self.vectors[indices]
        self.norm_vectors = self.norm_vectors[indices]
        self.items = {w: idx for idx, w in enumerate(wordlist)}
        self.indices = {v: k for k, v in self.items.items()}
        if self.unk_index is not None:
            self.unk_index = self.items[wordlist[self.unk_index]]