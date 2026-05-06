def get_next_holiday(self, division=None, date=None):
        """
        Returns the next known bank holiday
        :param division: see division constants; defaults to common holidays
        :param date: search starting from this date; defaults to today
        :return: dict
        """
        date = date or datetime.date.today()
        for holiday in self.get_holidays(division=division):
            if holiday['date'] > date:
                return holiday