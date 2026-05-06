def words(self):
        """Return a list of word tokens. This excludes punctuation characters.
        If you want to include punctuation characters, access the ``tokens``
        property.

        :returns: A :class:`WordList <WordList>` of word tokens.

        """
        return WordList(
            word_tokenize(self.raw, self.tokenizer, include_punc=False))