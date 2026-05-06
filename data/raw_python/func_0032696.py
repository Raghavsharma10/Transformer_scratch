def setrange(self, name, offset, value):
        """
        Overwrite bytes in the value of ``name`` starting at ``offset`` with
        ``value``. If ``offset`` plus the length of ``value`` exceeds the
        length of the original value, the new value will be larger
        than before.
        If ``offset`` exceeds the length of the original value, null bytes
        will be used to pad between the end of the previous value and the start
        of what's being injected.

        Returns the length of the new string.
        :param name: str     the name of the redis key
        :param offset: int
        :param value: str
        :return: Future()
        """
        with self.pipe as pipe:
            return pipe.setrange(self.redis_key(name), offset, value)