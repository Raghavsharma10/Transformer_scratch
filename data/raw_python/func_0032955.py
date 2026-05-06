def decode(cls, value):
        """
        decode the data from redis.
        :param value: bytes
        :return: list
        """
        try:
            data = [v for v in value.decode(cls._encoding).split(',') if
                    v != '']
            return data if data else None
        except AttributeError:
            return value