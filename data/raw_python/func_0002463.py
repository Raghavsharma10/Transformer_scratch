def lists(self, column, key=None):
        """
        Get a list with the values of a given column

        :param column: The column to get the values for
        :type column: str

        :param key: The key
        :type key: str

        :return: The list of values
        :rtype: list or dict
        """
        results = self._query.lists(column, key)

        if self._model.has_get_mutator(column):
            if isinstance(results, dict):
                for key, value in results.items():
                    fill = {column: value}

                    results[key] = self._model.new_from_builder(fill).column
            else:
                for i, value in enumerate(results):
                    fill = {column: value}

                    results[i] = self._model.new_from_builder(fill).column

        return results