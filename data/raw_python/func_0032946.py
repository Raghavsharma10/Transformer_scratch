def encode(cls, value):
        """
        convert a boolean value into something we can persist to redis.
        An empty string is the representation for False.

        :param value: bool
        :return: bytes
        """
        if value not in [True, False]:
            raise InvalidValue('not a boolean')

        return b'1' if value else b''