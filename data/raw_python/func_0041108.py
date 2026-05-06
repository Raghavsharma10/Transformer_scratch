def add_hours(self, datetimestr, n):
        """Returns a time that n hours after a time.

        :param datetimestr: a datetime object or a datetime str
        :param n: number of hours, value can be negative

        **中文文档**

        返回给定日期N小时之后的时间。
        """
        a_datetime = self.parse_datetime(datetimestr)
        return a_datetime + timedelta(seconds=3600 * n)