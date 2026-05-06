def add_error(self, group, term, sub_term, value):
        """For records that are not defined as terms, either add it to the
        errors list."""

        self._errors[(group, term, sub_term)] = value