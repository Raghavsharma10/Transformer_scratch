def _from_to_as_term(self, frm, to):
        """ Turns from and to into the query format.

        Args:
            frm (str): from year
            to (str): to year

        Returns:
            FTS query str with years range.

        """

        # The wackiness with the conversion to int and str, and adding ' ', is because there
        # can't be a space between the 'TO' and the brackets in the time range
        # when one end is open
        from_year = ''
        to_year = ''

        def year_or_empty(prefix, year, suffix):
            try:
                return prefix + str(int(year)) + suffix
            except (ValueError, TypeError):
                return ''

        if frm:
            from_year = year_or_empty('', frm, ' ')

        if to:
            to_year = year_or_empty(' ', to, '')

        if bool(from_year) or bool(to_year):
            return '[{}TO{}]'.format(from_year, to_year)
        else:
            return None