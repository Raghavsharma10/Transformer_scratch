def _get_columns(self, blueprint):
        """
        Get the blueprint's columns definitions.

        :param blueprint: The blueprint
        :type blueprint: Blueprint

        :rtype: list
        """
        columns = []

        for column in blueprint.get_added_columns():
            sql = self.wrap(column) + ' ' + self._get_type(column)

            columns.append(self._add_modifiers(sql, blueprint, column))

        return columns