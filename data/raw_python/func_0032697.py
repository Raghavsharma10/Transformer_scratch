def setbit(self, name, offset, value):
        """
        Flag the ``offset`` in the key as ``value``. Returns a boolean
        indicating the previous value of ``offset``.

        :param name: str     the name of the redis key
        :param  offset: int
        :param value:
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.setbit(self.redis_key(name), offset, value)