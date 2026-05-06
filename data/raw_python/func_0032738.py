def _value_decode(cls, member, value):
        """
        Internal method used to decode values from redis hash

        :param member: str
        :param value: bytes
        :return: multi
        """
        if value is None:
            return None
        try:
            field_validator = cls.fields[member]
        except KeyError:
            return cls.valueparse.decode(value)

        return field_validator.decode(value)