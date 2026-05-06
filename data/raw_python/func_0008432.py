def tokenize(self, text, include_punc=True, nested=False):
        """Return a list of word tokens.

        :param text: string of text.
        :param include_punc: (optional) whether to include punctuation as separate
            tokens. Default to True.
        :param nested: (optional) whether to return tokens as nested lists of
            sentences. Default to False.

        """
        self.tokens = [
            w for w in (
                self.word_tokenize(
                    s,
                    include_punc) for s in self.sent_tokenize(text))]
        if nested:
            return self.tokens
        else:
            return list(chain.from_iterable(self.tokens))