def set(self, key, value):
        """Sets the specified key value.

        :param str|unicode key:

        :param int|str|unicode value:

        :rtype: bool
        """
        return uwsgi.cache_set(key, value, self.timeout, self.name)