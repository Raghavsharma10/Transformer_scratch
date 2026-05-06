def set_column_size_limit(self, column_name: str, size_limit: int):
        """
        Sets the size limit of a specific column.

        :param column_name: The name of the column to change.
        :param size_limit: The max size of the column width.
        """
        if self._column_size_map.get(column_name):
            self._column_size_map[column_name] = size_limit
        else:
            raise ValueError(f'There is no column named {column_name}!')