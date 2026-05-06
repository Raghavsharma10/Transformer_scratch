def pexpire(self, name, time):
        """
        Set an expire flag on key ``name`` for ``time`` milliseconds.
        ``time`` can be represented by an integer or a Python timedelta
        object.

        :param name: str
        :param time: int or timedelta
        :return Future
        """
        with self.pipe as pipe:
            return pipe.pexpire(self.redis_key(name), time)