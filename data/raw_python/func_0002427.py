def delete(self, id=None):
        """
        Delete a record from the database

        :param id: The id of the row to delete
        :type id: mixed

        :return: The number of rows deleted
        :rtype: int
        """
        if id is not None:
            self.where('id', '=', id)

        sql = self._grammar.compile_delete(self)

        return self._connection.delete(sql, self.get_bindings())