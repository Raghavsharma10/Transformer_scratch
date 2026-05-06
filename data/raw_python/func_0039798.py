def random(self, length=4):
        """Method to randomize the DigitWord to a given length; for example obj.random(length=4) would
        produce a DigitWord containing of four random Digits or HexDigits. The type of digit created
        is set by the wordtype."""
        if not isinstance(length, int):
            raise TypeError('DigitWord can only be randomized by an integer length')

        if self.wordtype == DigitWord.DIGIT:
            self._word = [Digit(random.randint(0, 9)) for i in range(0, length)]
        elif self.wordtype == DigitWord.HEXDIGIT:
            self._word = [HexDigit(str(hex(random.randint(0, 15))).replace('0x',''))
                          for i in range(0, length)]
        else:
            raise TypeError('wordtype is invalid.')