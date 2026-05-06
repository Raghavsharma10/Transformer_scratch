def weekday_series(self, start, end, weekday, return_date=False):
        """Generate a datetime series with same weekday number.

        ISO weekday number: Mon to Sun = 1 to 7

        Usage::

            >>> start, end = "2014-01-01 06:30:25", "2014-02-01 06:30:25"
            >>> rolex.weekday_series(start, end, weekday=2) # All Tuesday
            [
                datetime(2014, 1, 7, 6, 30, 25),
                datetime(2014, 1, 14, 6, 30, 25),
                datetime(2014, 1, 21, 6, 30, 25),
                datetime(2014, 1, 28, 6, 30, 25),
            ]

        :param weekday: int or list of int

        **中文文档**

        生成星期数一致的时间序列。
        """
        start = self.parse_datetime(start)
        end = self.parse_datetime(end)

        if isinstance(weekday, integer_types):
            weekday = [weekday, ]

        series = list()
        for i in self.time_series(
                start, end, freq="1day", return_date=return_date):
            if i.isoweekday() in weekday:
                series.append(i)

        return series