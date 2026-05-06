def encode(cls, value):
        """
        take a list of strings and turn it into utf-8 byte-string

        :param value:
        :return:
        """
        coerced = unicode(value)
        if coerced == value and cls.PATTERN.match(coerced):
            return coerced.encode(cls._encoding)

        raise InvalidValue('not ascii')