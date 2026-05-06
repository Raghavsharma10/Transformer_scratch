def assignrepr(self, prefix, style=None, utcoffset=None):
        """Return a |repr| string with an prefixed assignement.

        Without option arguments given, printing the returned string
        looks like:

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('1996-11-01 00:00:00',
        ...                     '1997-11-01 00:00:00',
        ...                     '1d')
        >>> print(timegrid.assignrepr(prefix='timegrid = '))
        timegrid = Timegrid('1996-11-01 00:00:00',
                            '1997-11-01 00:00:00',
                            '1d')

        The optional arguments are passed to method |Date.to_repr|
        without any modifications:

        >>> print(timegrid.assignrepr(
        ...     prefix='', style='iso1', utcoffset=120))
        Timegrid('1996-11-01T01:00:00+02:00',
                 '1997-11-01T01:00:00+02:00',
                 '1d')
        """
        skip = len(prefix) + 9
        blanks = ' ' * skip
        return (f"{prefix}Timegrid('"
                f"{self.firstdate.to_string(style, utcoffset)}',\n"
                f"{blanks}'{self.lastdate.to_string(style, utcoffset)}',\n"
                f"{blanks}'{str(self.stepsize)}')")