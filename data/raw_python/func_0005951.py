def decr(self, key, delta=1):
        """Decrements the specified key value by the specified value.

        :param str|unicode key:

        :param int delta:

        :rtype: bool
        """
        return uwsgi.cache_dec(key, delta, self.timeout, self.name)