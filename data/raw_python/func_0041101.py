def rnd_date(self, start=date(1970, 1, 1), end=date.today()):
        """Generate a random date between ``start`` to ``end``.

        :param start: Left bound
        :type start: string or datetime.date, (default date(1970, 1, 1))
        :param end: Right bound
        :type end: string or datetime.date, (default date.today())
        :return: a datetime.date object

        **中文文档**

        随机生成一个位于 ``start`` 和 ``end`` 之间的日期。
        """
        if isinstance(start, string_types):
            start = self.str2date(start)
        if isinstance(end, string_types):
            end = self.str2date(end)
        if start > end:
            raise ValueError("start time has to be earlier than end time")
        return date.fromordinal(
            random.randint(start.toordinal(), end.toordinal()))