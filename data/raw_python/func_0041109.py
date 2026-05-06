def add_weeks(self, datetimestr, n, return_date=False):
        """Returns a time that n weeks after a time.

        :param datetimestr: a datetime object or a datetime str
        :param n: number of weeks, value can be negative
        :param return_date: returns a date object instead of datetime

        **中文文档**

        返回给定日期N周之后的时间。
        """
        a_datetime = self.parse_datetime(datetimestr)
        a_datetime += timedelta(days=7 * n)
        if return_date:
            return a_datetime.date()
        else:
            return a_datetime