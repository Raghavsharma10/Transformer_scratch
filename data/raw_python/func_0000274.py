def value_to_db(self, value):
        """ Returns field's single value prepared for saving into a database. """
        assert isinstance(value, six.integer_types)
        return str(value).encode("utf_8")