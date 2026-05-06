def delete(self, table):
        """Deletes record in table
        >>> yql.delete('yql.storage').where(['name','=','store://YEl70PraLLMSMuYAauqNc7'])
        """
        self._table = table
        self._limit = None
        self._query = "DELETE FROM {0}".format(self._table)

        return self