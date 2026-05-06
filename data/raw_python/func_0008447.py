def tokenize(self, tokenizer=None):
        """Return a list of tokens, using ``tokenizer``.

        :param tokenizer: (optional) A tokenizer object. If None, defaults to
            this blob's default tokenizer.

        """
        t = tokenizer if tokenizer is not None else self.tokenizer
        return WordList(t.tokenize(self.raw))