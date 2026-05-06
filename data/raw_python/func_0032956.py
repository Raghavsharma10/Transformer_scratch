def encode(cls, value):
        """
        the list it so it can be stored in redis.

        :param value: list
        :return: bytes
        """
        try:
            coerced = [str(v) for v in value]
            if coerced == value:
                return ",".join(coerced).encode(cls._encoding) if len(
                    value) > 0 else None
        except TypeError:
            pass

        raise InvalidValue('not a list of strings')