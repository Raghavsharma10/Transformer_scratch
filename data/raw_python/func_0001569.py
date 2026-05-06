def tokenize(self, string):
        """
        Tokenizer utility.
        When processing byte, outputs the string unaltered.
        The character unit type is used for unicode data, the string is
        decoded according to the encoding provided.
        In the case of word unit, EOL characters are detached from the
        preceding word, and outputs the list of words, i.e. the list of non-space strings
        separated by space strings.


        >>> SA=SuffixArray('abecedaire', UNIT_BYTE)

        >>> SA.tokenize('abecedaire')=='abecedaire'
        True
        >>> len(SA.tokenize('abecedaire'))
        10

        >>> SA=SuffixArray('abecedaire', UNIT_BYTE, "utf-8")

        >>> SA.tokenize('abecedaire')==u'abecedaire'
        True
        >>> len(SA.tokenize('abecedaire'))
        10

        >>> SA=SuffixArray('mississippi', UNIT_WORD)

        >>> SA.tokenize('miss issi ppi')
        ['miss', 'issi', 'ppi']

        >>> SA.tokenize('miss issi\\nppi')
        ['miss', 'issi', '\\n', 'ppi']

        """
        if self.unit == UNIT_WORD:
            # the EOL character is treated as a word, hence a substitution
            # before split

            return [token for token in string.replace("\n", " \n ").split(self.tokSep) if token != ""]
        elif self.unit == UNIT_CHARACTER:
            return string.decode(self.encoding)
        else:
            return string