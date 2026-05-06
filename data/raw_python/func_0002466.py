def get_models(self, columns=None):
        """
        Get the hydrated models without eager loading.

        :param columns: The columns to get
        :type columns: list

        :return: A list of models
        :rtype: list
        """
        results = self._query.get(columns)

        connection = self._model.get_connection_name()

        return self._model.hydrate(results, connection).all()