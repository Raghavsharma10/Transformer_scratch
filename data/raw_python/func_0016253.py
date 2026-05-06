def _raw_sql(self, values):
        """Prepare SQL statement consisting of a sequence of WHEN .. THEN statements."""
        if isinstance(self.model._meta.pk, CharField):
            when_clauses = " ".join(
                [self._when("'{}'".format(x), y) for (x, y) in values]
            )
        else:
            when_clauses = " ".join([self._when(x, y) for (x, y) in values])
        table_name = self.model._meta.db_table
        primary_key = self.model._meta.pk.column
        return 'SELECT CASE {}."{}" {} ELSE 0 END'.format(
            table_name, primary_key, when_clauses
        )