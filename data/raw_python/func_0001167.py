def strip_callables(row):
        """Extract callable values from `row`.

        Replace the callable values with the initial value (if specified) or
        an empty string.

        Parameters
        ----------
        row : mapping
            A data row.  The keys are either a single column name or a tuple of
            column names.  The values take one of three forms: 1) a
            non-callable value, 2) a tuple (initial_value, callable), 3) or a
            single callable (in which case the initial value is set to an empty
            string).

        Returns
        -------
        list of (column, callable)
        """
        callables = []
        to_delete = []
        to_add = []
        for columns, value in row.items():
            if isinstance(value, tuple):
                initial, fn = value
            else:
                initial = NOTHING
                # Value could be a normal (non-callable) value or a
                # callable with no initial value.
                fn = value

            if callable(fn) or inspect.isgenerator(fn):
                lgr.debug("Using %r as the initial value "
                          "for columns %r in row %r",
                          initial, columns, row)
                if not isinstance(columns, tuple):
                    columns = columns,
                else:
                    to_delete.append(columns)
                for column in columns:
                    to_add.append((column, initial))
                callables.append((columns, fn))

        for column, value in to_add:
            row[column] = value
        for multi_columns in to_delete:
            del row[multi_columns]

        return callables