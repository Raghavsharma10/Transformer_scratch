def _add_column(self, type, name, **parameters):
        """
        Add a new column to the blueprint.

        :param type: The column type
        :type type: str

        :param name: The column name
        :type name: str

        :param parameters: The column parameters
        :type parameters: dict

        :rtype: Fluent
        """
        parameters.update({
            'type': type,
            'name': name
        })

        column = Fluent(**parameters)
        self._columns.append(column)

        return column