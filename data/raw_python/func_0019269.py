def from_cfunits(cls, units) -> 'Date':
        """Return a |Date| object representing the reference date of the
        given `units` string agreeing with the NetCDF-CF conventions.

        The following example string is taken from the `Time Coordinate`_
        chapter of the NetCDF-CF conventions documentation (modified).
        Note that the first entry (the unit) is ignored:

        >>> from hydpy import Date
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42 -6:00')
        Date('1992-10-08 22:15:42')
        >>> Date.from_cfunits(' day since 1992-10-8 15:15:00')
        Date('1992-10-08 15:15:00')
        >>> Date.from_cfunits('seconds since 1992-10-8 -6:00')
        Date('1992-10-08 07:00:00')
        >>> Date.from_cfunits('m since 1992-10-8')
        Date('1992-10-08 00:00:00')

        Without modification, when "0" is included as the decimal fractions
        of a second, the example string from `Time Coordinate`_ can also
        be passed.  However, fractions different from "0" result in
        an error:

        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.')
        Date('1992-10-08 15:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.00')
        Date('1992-10-08 15:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42. -6:00')
        Date('1992-10-08 22:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.0 -6:00')
        Date('1992-10-08 22:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.005 -6:00')
        Traceback (most recent call last):
        ...
        ValueError: While trying to parse the date of the NetCDF-CF "units" \
string `seconds since 1992-10-8 15:15:42.005 -6:00`, the following error \
occurred: No other decimal fraction of a second than "0" allowed.
        """
        try:
            string = units[units.find('since')+6:]
            idx = string.find('.')
            if idx != -1:
                jdx = None
                for jdx, char in enumerate(string[idx+1:]):
                    if not char.isnumeric():
                        break
                    if char != '0':
                        raise ValueError(
                            'No other decimal fraction of a second '
                            'than "0" allowed.')
                else:
                    if jdx is None:
                        jdx = idx+1
                    else:
                        jdx += 1
                string = f'{string[:idx]}{string[idx+jdx+1:]}'
            return cls(string)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to parse the date of the NetCDF-CF "units" '
                f'string `{units}`')