def encode(cls, value):
        """
        write binary data into redis without encoding it.

        :param value: bytes
        :return: bytes
        """
        try:
            coerced = bytes(value)
            if coerced == value:
                return coerced
        except (TypeError, UnicodeError):
            pass

        raise InvalidValue('not binary')