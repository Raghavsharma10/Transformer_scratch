def add_minutes(self, datetimestr, n):
        """Returns a time that n minutes after a time.

        :param datetimestr: a datetime object or a datetime str
        :param n: number of minutes, value can be negative

        **中文文档**

        返回给定日期N分钟之后的时间。
        """
        a_datetime = self.parse_datetime(datetimestr)
        return a_datetime + timedelta(seconds=60 * n)