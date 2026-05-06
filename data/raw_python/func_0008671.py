def get_db_prep_value(self, value, connection, prepared=False):
        """Convert a value to DB storage.

        Returns the state name.
        """
        if not prepared:
            value = self.get_prep_value(value)
        return value.state.name