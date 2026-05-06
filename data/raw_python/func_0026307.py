def get_next_work_day(self, division=None, date=None):
        """
        Returns the next work day, skipping weekends and bank holidays
        :param division: see division constants; defaults to common holidays
        :param date: search starting from this date; defaults to today
        :return: datetime.date; NB: get_next_holiday returns a dict
        """
        date = date or datetime.date.today()
        one_day = datetime.timedelta(days=1)
        holidays = set(holiday['date'] for holiday in self.get_holidays(division=division))
        while True:
            date += one_day
            if date.weekday() not in self.weekend and date not in holidays:
                return date