def add_years(self, datetimestr, n, return_date=False):
        """Returns a time that n years after a time.

        :param datetimestr: a datetime object or a datetime str
        :param n: number of years, value can be negative
        :param return_date: returns a date object instead of datetime

        **中文文档**

        返回给定日期N年之后的时间。
        """
        a_datetime = self.parse_datetime(datetimestr)

        try:
            a_datetime = datetime(
                a_datetime.year + n, a_datetime.month, a_datetime.day,
                a_datetime.hour, a_datetime.minute,
                a_datetime.second, a_datetime.microsecond)
        except:
            a_datetime = datetime(
                a_datetime.year + n, 2, 28,
                a_datetime.hour, a_datetime.minute,
                a_datetime.second, a_datetime.microsecond)

        if return_date:
            return a_datetime.date()
        else:
            return a_datetime