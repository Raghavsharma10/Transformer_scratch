def add_set_membership(self, values, column_name):
        """
        Append a set membership test, creating a query of the form 'WHERE name IN (?,?...?)'.

        :param values:
            A list of values, or a subclass of basestring. If this is non-None and non-empty this will add a set
            membership test to the state. If the supplied value is a basestring it will be wrapped in a single element
            list. Values are mapped by SQLBuilder._map_value before being added, so e.g. NSString instances will work
            here.
        :param column_name:
            The name of the column to use when checking the 'IN' condition.
        """
        if values is not None and len(values) > 0:
            if isinstance(values, basestring):
                values = [values]
            question_marks = ', '.join(["%s"] * len(values))
            self.where_clauses.append('{0} IN ({1})'.format(column_name, question_marks))
            for value in values:
                self.sql_args.append(SQLBuilder.map_value(value))