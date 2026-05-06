def encode(cls, value):
        """
        take an integer and turn it into a string representation
        to write into redis.

        :param value: int
        :return: str
        """
        try:
            coerced = int(value)
            if coerced + 0 == value:
                return repr(coerced)

        except (TypeError, ValueError):
            pass

        raise InvalidValue('not an int')