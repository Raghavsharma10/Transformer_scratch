def from_utctimestamp(self, timestamp):
        """Create a **UTC datetime** object that number of seconds after
        UTC 1970-01-01 00:00:00. If you want local time, use
        :meth:`Rolex.from_timestamp`

        Because python doesn't support negative timestamp to datetime
        so we have to implement my own method.

        **中文文档**

        返回一个在UTC 1970-01-01 00:00:00 之后 #timestamp 秒后的时间。默认为
        UTC时间。即返回的datetime不带tzinfo
        """
        if timestamp >= 0:
            return datetime.utcfromtimestamp(timestamp)
        else:
            return datetime(1970, 1, 1) + timedelta(seconds=timestamp)