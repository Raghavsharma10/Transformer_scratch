def to_cfunits(self, unit='hours', utcoffset=None):
        """Return a `units` string agreeing with the NetCDF-CF conventions.

        By default, |Date.to_cfunits| takes `hours` as time unit, and the
        the actual value of |Options.utcoffset| as time zone information:

        >>> from hydpy import Date
        >>> date = Date('1992-10-08 15:15:42')
        >>> date.to_cfunits()
        'hours since 1992-10-08 15:15:42 +01:00'

        Other time units are allowed (no checks are performed, so select
        something useful):

        >>> date.to_cfunits(unit='minutes')
        'minutes since 1992-10-08 15:15:42 +01:00'

        For changing the time zone, pass the corresponding offset in minutes:

        >>> date.to_cfunits(unit='sec', utcoffset=-60)
        'sec since 1992-10-08 13:15:42 -01:00'
        """
        if utcoffset is None:
            utcoffset = hydpy.pub.options.utcoffset
        string = self.to_string('iso2', utcoffset)
        string = ' '.join((string[:-6], string[-6:]))
        return f'{unit} since {string}'