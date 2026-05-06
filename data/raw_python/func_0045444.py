def populate_values(self, rows, field):
        """
        Populates the filter values of this filter using list of rows.

        :param list[dict[str,T]] rows: The row set.
        :param str field: The field name.
        """
        self._values.clear()
        for row in rows:
            condition = SimpleConditionFactory.create_condition(self._field, row[field])
            if condition.scheme == 'plain':
                self._values.append(condition.expression)
            else:
                self._conditions.append(condition)