def to_string(self, style=None, utcoffset=None):
        """Return a |str| object representing the actual date in
        accordance with the given style and the eventually given
        UTC offset (in minutes).

        Without any input arguments, the actual |Date.style| is used
        to return a date string in your local time zone:

        >>> from hydpy import Date
        >>> date = Date('01.11.1997 00:00:00')
        >>> date.to_string()
        '01.11.1997 00:00:00'

        Passing a style string affects the returned |str| object, but
        not the |Date.style| property:

        >>> date.style
        'din1'
        >>> date.to_string(style='iso2')
        '1997-11-01 00:00:00'
        >>> date.style
        'din1'

        When passing the `utcoffset` in minutes, the offset string is
        appended:

        >>> date.to_string(style='iso2', utcoffset=60)
        '1997-11-01 00:00:00+01:00'

        If the given offset does not correspond to your local offset
        defined by |Options.utcoffset| (which defaults to UTC+01:00),
        the date string is adapted:

        >>> date.to_string(style='iso1', utcoffset=0)
        '1997-10-31T23:00:00+00:00'
        """
        if not style:
            style = self.style
        if utcoffset is None:
            string = ''
            date = self.datetime
        else:
            sign = '+' if utcoffset >= 0 else '-'
            hours = abs(utcoffset // 60)
            minutes = abs(utcoffset % 60)
            string = f'{sign}{hours:02d}:{minutes:02d}'
            offset = utcoffset-hydpy.pub.options.utcoffset
            date = self.datetime + datetime.timedelta(minutes=offset)
        return date.strftime(self._formatstrings[style]) + string