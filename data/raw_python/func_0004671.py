def delete_columns(self, columns):
        """
        Delete columns from the DataFrame

        :param columns: list of columns to delete
        :return: nothing
        """
        columns = [columns] if not isinstance(columns, (list, blist)) else columns
        if not all([x in self._columns for x in columns]):
            raise ValueError('all columns must be in current columns')
        for column in columns:
            c = self._columns.index(column)
            del self._data[c]
            del self._columns[c]
        if not len(self._data):  # if all the columns have been deleted, remove index
            self.index = list()