def _create_indices(self, table_name, indices):
        """
        Create multiple indices (each over multiple columns) on a given table.

        Parameters
        ----------
        table_name : str

        indices : iterable of tuples
            Multiple groups of columns, each of which should be indexed.
        """
        require_string(table_name, "table_name")
        require_iterable_of(indices, (tuple, list))
        for index_column_set in indices:
            self._create_index(table_name, index_column_set)