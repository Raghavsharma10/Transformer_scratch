def rename_columns(self, rename_dict):
        """
        Renames the columns

        :param rename_dict: dict where the keys are the current column names and the values are the new names
        :return: nothing
        """
        if not all([x in self._columns for x in rename_dict.keys()]):
            raise ValueError('all dictionary keys must be in current columns')
        for current in rename_dict.keys():
            self._columns[self._columns.index(current)] = rename_dict[current]