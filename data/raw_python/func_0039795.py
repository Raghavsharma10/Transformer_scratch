def word(self):
        """Property of the DigitWord returning (or setting) the DigitWord as a list of integers (or
        string representations) of DigitModel. The property is called during instantiation as the
        property validates the value passed and ensures that all digits are valid."""

        if self.wordtype == DigitWord.DIGIT:
            return self._word
        else:
            # Strip out '0x' from the string representation. Note, this could be replaced with the
            # following code: str(hex(a))[2:] but is more obvious in the code below.
            return [str(hex(a)).replace('0x', '') for a in self._word]