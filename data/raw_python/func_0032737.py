def _value_encode(cls, member, value):
        """
        Internal method used to encode values into the hash.

        :param member: str
        :param value: multi
        :return: bytes
        """
        try:
            field_validator = cls.fields[member]
        except KeyError:
            return cls.valueparse.encode(value)

        return field_validator.encode(value)