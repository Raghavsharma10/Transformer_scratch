def xform(self, number, base):
        """
        Get a number as a string.

        :param number: a number
        :type number: list of int
        :param int base: the base in which this number is being represented
        :raises BasesValueError: if config is unsuitable for number
        """
        if self.CONFIG.use_letters:
            digits = \
               self._UPPER_DIGITS if self.CONFIG.use_caps else \
               self._LOWER_DIGITS
            return ''.join(digits[x] for x in number)
        separator = '' if base <= 10 else self.CONFIG.separator
        return separator.join(str(x) for x in number)