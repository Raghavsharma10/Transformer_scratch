def index(self, item):
        """
        Return the 0-based position of `item` in this IpRange.


        >>> r = IpRange('127.0.0.1', '127.255.255.255')
        >>> r.index('127.0.0.1')
        0
        >>> r.index('127.255.255.255')
        16777214
        >>> r.index('10.0.0.1')
        Traceback (most recent call last):
            ...
        ValueError: 10.0.0.1 is not in range


        :param item: Dotted-quad ip address.
        :type item: str
        :returns: Index of ip address in range
        """
        item = self._cast(item)
        offset = item - self.startIp
        if offset >= 0 and offset < self._len:
            return offset
        raise ValueError('%s is not in range' % self._ipver.long2ip(item))