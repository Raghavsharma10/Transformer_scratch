def value_to_db(self, value):
        """ Returns field's single value prepared for saving into a database. """
        assert isinstance(value, datetime.date)
        assert not isinstance(value, datetime.datetime)

        try:
            value = value - datetime.date(year=1970, month=1, day=1)
        except OverflowError:
            raise tldap.exceptions.ValidationError("is too big a date")

        return str(value.days).encode("utf_8")