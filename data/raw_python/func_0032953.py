def decode(cls, value):
        """
        take a utf-8 encoded byte-string from redis and
        turn it back into a list

        :param value: bytes
        :return: list
        """
        try:
            return None if value is None else \
                list(json.loads(value.decode(cls._encoding)))
        except (TypeError, AttributeError):
            return list(value)