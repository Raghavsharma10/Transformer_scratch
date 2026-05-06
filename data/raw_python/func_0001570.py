def reprString(self, string, length):
        """
        Output a string of length tokens in the original form.
        If string is an integer, it is considered as an offset in the text.
        Otherwise string is considered as a sequence of ids (see voc and
        tokId).

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.reprString(0, 3)
        'mis'

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.reprString([1, 4, 1, 3, 3, 2], 5)
        'isipp'

        >>> SA=SuffixArray('missi ssi ppi', UNIT_WORD)
        >>> SA.reprString(0, 3)
        'missi ssi ppi'

        >>> SA=SuffixArray('missi ssi ppi', UNIT_WORD)
        >>> SA.reprString([1, 3, 2], 3)
        'missi ssi ppi'
        """
        if isinstance(string, int):
            length = min(length, self.length - string)
            string = self.string[string:string + length]

        voc = self.voc
        res = self.tokSep.join((voc[id] for id in string[:length]))
        if self.unit == UNIT_WORD:
            res = res.replace(" \n", "\n")
            res = res.replace("\n ", "\n")

        if self.unit == UNIT_CHARACTER:
            res = res.encode(self.encoding)

        return res