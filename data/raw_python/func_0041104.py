def rnd_datetime(self, start=datetime(1970, 1, 1), end=datetime.now()):
        """Generate a random datetime between ``start`` to ``end``.

        :param start: Left bound
        :type start: string or datetime.datetime, (default datetime(1970, 1, 1))
        :param end: Right bound
        :type end: string or datetime.datetime, (default datetime.now())
        :return: a datetime.datetime object

        **中文文档**

        随机生成一个位于 ``start`` 和 ``end`` 之间的时间。
        """
        if isinstance(start, string_types):
            start = self.str2datetime(start)
        if isinstance(end, str):
            end = self.str2datetime(end)
        if start > end:
            raise ValueError("start time has to be earlier than end time")
        return self.from_utctimestamp(
            random.randint(
                int(self.to_utctimestamp(start)),
                int(self.to_utctimestamp(end)),
            )
        )