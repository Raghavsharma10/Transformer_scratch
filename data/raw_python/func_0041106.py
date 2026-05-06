def add_seconds(self, datetimestr, n):
        """Returns a time that n seconds after a time.

        :param datetimestr: a datetime object or a datetime str
        :param n: number of seconds, value can be negative

        **中文文档**

        返回给定日期N秒之后的时间。
        """
        a_datetime = self.parse_datetime(datetimestr)
        return a_datetime + timedelta(seconds=n)