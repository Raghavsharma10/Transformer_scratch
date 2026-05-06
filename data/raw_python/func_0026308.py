def is_work_day(self, date, division=None):
        """
        True if the date is not a weekend or a known bank holiday
        :param date: the date to check
        :param division: see division constants; defaults to common holidays
        :return: bool
        """
        return date.weekday() not in self.weekend and date not in (
            holiday['date'] for holiday in self.get_holidays(division=division)
        )