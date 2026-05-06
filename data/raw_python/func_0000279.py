def value_to_db(self, value):
        """ Returns field's single value prepared for saving into a database. """
        assert isinstance(value, datetime.datetime)

        try:
            value = value - datetime.datetime(1970, 1, 1)
        except OverflowError:
            raise tldap.exceptions.ValidationError("is too big a date")

        value = value.seconds + value.days * 24 * 3600
        value = str(value).encode("utf_8")

        return value