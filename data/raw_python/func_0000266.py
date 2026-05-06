def to_db(self, value):
        """ Returns field's single value prepared for saving into a database. """

        # ensure value is valid
        self.validate(value)

        assert isinstance(value, list)
        value = list(value)
        for i, v in enumerate(value):
            value[i] = self.value_to_db(v)

        # return result
        assert isinstance(value, list)
        return value