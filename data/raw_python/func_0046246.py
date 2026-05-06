def xform(self, radix, relation):
        """
        Transform a radix and some information to a str according to
        configurations.

        :param Radix radix: the radix
        :param int relation: relation of display value to actual value
        :param units: element of UNITS()
        :returns: a string representing the value
        :rtype: str

        :raises BasesValueError: if configuration does not work with value
        """
        right = radix.non_repeating_part
        left = radix.integer_part
        repeating = radix.repeating_part

        if repeating == []:
            right = self.STRIP.xform(right, relation)

        right_str = self.DIGITS.xform(right, radix.base)
        left_str = self.DIGITS.xform(left, radix.base) or '0'
        repeating_str = self.DIGITS.xform(repeating, radix.base)

        number = self.NUMBER.xform(
           left_str,
           right_str,
           repeating_str,
           radix.base,
           radix.sign
        )

        decorators = self.DECORATORS.decorators(relation)

        result = {
           'approx' : decorators.approx_str,
           'space' : ' ' if decorators.approx_str else '',
           'number' : number
        }

        return self._FMT_STR % result