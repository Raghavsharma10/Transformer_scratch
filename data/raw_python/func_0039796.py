def word(self, value):
        """Property of the DigitWord returning (or setting) the DigitWord as a list of integers (or
        string representations) of DigitModel. The property is called during instantiation as the
        property validates the value passed and ensures that all digits are valid. The values can
        be passed as ANY iterable"""

        self._validate_word(value=value)

        _word = []

        # Iterate the values passed.
        for a in value:
            # Check the value is an int or a string.
            if not (isinstance(a, int) or isinstance(a, str) or isinstance(a, unicode)):
                raise ValueError('DigitWords must be made from digits (strings or ints) '
                                 'between 0 and 9 for decimal and 0 and 15 for hex')

            # This convoluted check is caused by the remove of the unicode type in Python 3+
            # If this is Python2.x, then we need to convert unicode to string, otherwise
            # we leave it as is.
            if sys.version_info[0] == 2 and isinstance(a, unicode):
                _a = str(a)
            else:
                _a = a

            # Create the correct type of Digit based on the wordtype of the DigitWord
            if self.wordtype == DigitWord.DIGIT:
                _digit = Digit(_a)
            elif self.wordtype == DigitWord.HEXDIGIT:
                _digit = HexDigit(_a)
            else:
                raise TypeError('The wordtype is not valid.')

            _word.append(_digit)

        self._word = _word