def set(self, val, lifetime=None):
        """
        set cache value

        :param val: any picklable object
        :param lifetime: exprition time in sec
        :return: val
        """
        cache.set(self.key,
                  (json.dumps(val) if self.serialize else val),
                  lifetime or settings.DEFAULT_CACHE_EXPIRE_TIME)
        return val