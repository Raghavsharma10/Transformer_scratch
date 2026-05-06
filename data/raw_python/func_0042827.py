def add_sql(self, value, clause):
        """
        Add a WHERE clause to the state.

        :param value:
            The unknown to bind into the state. Uses SQLBuilder._map_value() to map this into an appropriate database
            compatible type.
        :param clause:
            A SQL fragment defining the restriction on the unknown value
        """
        if value is not None:
            self.sql_args.append(SQLBuilder.map_value(value))
            self.where_clauses.append(clause)