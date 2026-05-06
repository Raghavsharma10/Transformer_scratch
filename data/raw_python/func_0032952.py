def encode(cls, value):
        """
        take a list and turn it into a utf-8 encoded byte-string for redis.

        :param value: list
        :return: bytes
        """
        try:
            coerced = list(value)
            if coerced == value:
                return json.dumps(coerced).encode(cls._encoding)
        except TypeError:
            pass

        raise InvalidValue('not a list')