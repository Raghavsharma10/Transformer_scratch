def encode(cls, value):
        """
        encode the dict as a json string to be written into redis.

        :param value: dict
        :return: bytes
        """
        try:
            coerced = dict(value)
            if coerced == value:
                return json.dumps(coerced).encode(cls._encoding)
        except (TypeError, ValueError):
            pass
        raise InvalidValue('not a dict')