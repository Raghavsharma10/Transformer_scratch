def checkAlphabet(self, count=10):
        """
        A function which checks if an AA read really contains amino acids. This
        additional testing is needed, because the letters in the DNA alphabet
        are also in the AA alphabet.

        @param count: An C{int}, indicating how many bases or amino acids at
            the start of the sequence should be considered. If C{None}, all
            bases are checked.
        @return: C{True} if the alphabet characters in the first C{count}
            positions of sequence is a subset of the allowed alphabet for this
            read class, or if the read class has a C{None} alphabet.
        @raise ValueError: If a DNA sequence has been passed to AARead().
        """
        if six.PY3:
            readLetters = super().checkAlphabet(count)
        else:
            readLetters = Read.checkAlphabet(self, count)
        if len(self) > 10 and readLetters.issubset(set('ACGT')):
            raise ValueError('It looks like a DNA sequence has been passed to '
                             'AARead().')
        return readLetters