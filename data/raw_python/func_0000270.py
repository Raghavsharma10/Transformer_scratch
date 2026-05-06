def value_to_db(self, value):
        """ Returns field's single value prepared for saving into a database. """
        if isinstance(value, six.string_types):
            value = value.encode("utf_8")
        return value