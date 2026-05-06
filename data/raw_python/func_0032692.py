def psetex(self, name, value, time_ms):
        """
        Set the value of key ``name`` to ``value`` that expires in ``time_ms``
        milliseconds. ``time_ms`` can be represented by an integer or a Python
        timedelta object
        """
        with self.pipe as pipe:
            return pipe.psetex(self.redis_key(name), time_ms=time_ms,
                               value=self.valueparse.encode(value=value))