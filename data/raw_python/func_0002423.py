def first(self, limit=1, columns=None):
        """
        Execute the query and get the first results

        :param limit: The number of results to get
        :type limit: int

        :param columns: The columns to get
        :type columns: list

        :return: The result
        :rtype: mixed
        """
        if not columns:
            columns = ['*']

        results = self.take(limit).get(columns)

        if len(results) > 0:
            return results[0]

        return