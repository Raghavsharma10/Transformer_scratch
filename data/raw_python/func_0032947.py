def encode(cls, value):
        """
        encode a floating point number to bytes in redis

        :param value: float
        :return: bytes
        """
        try:
            if float(value) + 0 == value:
                return repr(value)
        except (TypeError, ValueError):
            pass

        raise InvalidValue('not a float')