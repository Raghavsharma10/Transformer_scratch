def tokenize(self, text, include_punc=True, **kwargs):
        """Return a list of word tokens.

        :param text: string of text.
        :param include_punc: (optional) whether to include punctuation as separate
            tokens. Default to True.

        """
        return self.tokenizer.word_tokenize(text, include_punc, **kwargs)