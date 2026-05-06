def get_holidays(self, division=None, year=None):
        """
        Gets a list of all known bank holidays, optionally filtered by division and/or year
        :param division: see division constants; defaults to common holidays
        :param year: defaults to all available years
        :return: list of dicts with titles, dates, etc
        """
        if division:
            holidays = self.data[division]
        else:
            holidays = self.data[self.ENGLAND_AND_WALES]
            dates_in_common = six.moves.reduce(
                set.intersection,
                (set(map(lambda holiday: holiday['date'], division_holidays))
                 for division, division_holidays in six.iteritems(self.data))
            )
            holidays = filter(lambda holiday: holiday['date'] in dates_in_common, holidays)
        if year:
            holidays = filter(lambda holiday: holiday['date'].year == year, holidays)
        return list(holidays)