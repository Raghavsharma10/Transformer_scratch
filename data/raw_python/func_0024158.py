def get(self, default=None):
        """
        return the cached value or default if it can't be found

        :param default: default value
        :return: cached value
        """
        d = cache.get(self.key)
        return ((json.loads(d.decode('utf-8')) if self.serialize else d)
                if d is not None
                else default)