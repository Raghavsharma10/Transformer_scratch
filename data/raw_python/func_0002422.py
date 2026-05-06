def find(self, id, columns=None):
        """
        Execute a query for a single record by id

        :param id: The id of the record to retrieve
        :type id: mixed

        :param columns: The columns of the record to retrive
        :type columns: list

        :return: mixed
        :rtype: mixed
        """
        if not columns:
            columns = ['*']

        return self.where('id', '=', id).first(1, columns)