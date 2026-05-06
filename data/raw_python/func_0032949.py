def encode(cls, value):
        """
        take a valid unicode string and turn it into utf-8 bytes

        :param value: unicode, str
        :return: bytes
        """
        coerced = unicode(value)
        if coerced == value:
            return coerced.encode(cls._encoding)

        raise InvalidValue('not text')