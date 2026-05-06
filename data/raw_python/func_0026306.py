def is_holiday(self, date, division=None):
        """
        True if the date is a known bank holiday
        :param date: the date to check
        :param division: see division constants; defaults to common holidays
        :return: bool
        """
        return date in (holiday['date'] for holiday in self.get_holidays(division=division))