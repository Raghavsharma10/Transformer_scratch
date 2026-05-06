def pexpireat(self, name, when):
        """
        Set an expire flag on key ``name``. ``when`` can be represented
        as an integer representing unix time in milliseconds (unix time * 1000)
        or a Python datetime object.
        """
        with self.pipe as pipe:
            return pipe.pexpireat(self.redis_key(name), when)