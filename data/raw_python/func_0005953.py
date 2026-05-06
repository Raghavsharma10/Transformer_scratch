def div(self, key, value=2):
        """Divides the specified key value by the specified value.

        :param str|unicode key:

        :param int value:

        :rtype: bool
        """
        return uwsgi.cache_mul(key, value, self.timeout, self.name)