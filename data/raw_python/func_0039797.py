def load(self, value):
        """Load the value of the DigitWord from a JSON representation of a list. The representation is
        validated to be a string and the encoded data a list. The list is then validated to ensure each
        digit is a valid digit"""

        if not isinstance(value, str):
            raise TypeError('Expected JSON string')

        _value = json.loads(value)
        self._validate_word(value=_value)
        self.word = _value