def tostring(self: 'ErrorValue', extra_digits: int = 0, plusminus: str = ' +/- ', fmt: str = None) -> str:
        """Make a string representation of the value and its uncertainty.

        Inputs:
        -------
            ``extra_digits``: integer
                how many extra digits should be shown (plus or minus, zero means
                that the number of digits should be defined by the magnitude of
                the uncertainty).
            ``plusminus``: string
                the character sequence to be inserted in place of '+/-'
                including delimiting whitespace.
            ``fmt``: string or None
                how to format the output. Currently only strings ending in 'tex'
                are supported, which render ascii-exponentials (i.e. 3.1415e-2)
                into a format which is more appropriate to TeX.

        Outputs:
        --------
            the string representation.
        """
        if isinstance(fmt, str) and fmt.lower().endswith('tex'):
            return re.subn('(\d*)(\.(\d)*)?[eE]([+-]?\d+)',
                           lambda m: (r'$%s%s\cdot 10^{%s}$' % (m.group(1), m.group(2), m.group(4))).replace('None',
                                                                                                             ''),
                           self.tostring(extra_digits=extra_digits, plusminus=plusminus, fmt=None))[0]
        if isinstance(self.val, numbers.Real):
            try:
                Ndigits = -int(math.floor(math.log10(self.err))) + extra_digits
            except (OverflowError, ValueError):
                return str(self.val) + plusminus + str(self.err)
            else:
                return str(round(self.val, Ndigits)) + plusminus + str(round(self.err, Ndigits))
        return str(self.val) + ' +/- ' + str(self.err)