def to_utctimestamp(self, dt):
        """Calculate number of seconds from UTC 1970-01-01 00:00:00.

        When:

        - dt doesn't have tzinfo: assume it's a utc time
        - dt has tzinfo: use tzinfo

        WARNING, if your datetime object doens't have ``tzinfo``, make sure
        it's a UTC time, but **NOT a LOCAL TIME**.

        **中文文档**

        计算时间戳

        若:

        - 不带tzinfo: 则默认为是UTC time
        - 带tzinfo: 使用tzinfo
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=utc)
        delta = dt - datetime(1970, 1, 1, tzinfo=utc)
        return delta.total_seconds()