def where(self, where_string, **kwargs):
        """
        Select from a given Table or Column with the specified WHERE clause
        string. Additional keywords are passed to ExploreSqlDB.query(). For
        convenience, if there is no '=', '>', '<', 'like', or 'LIKE' clause
        in the WHERE statement .where() tries to match the input string
        against the primary key column of the Table.

        Args:
            where_string (str): Where clause for the query against the Table
            or Column

        Kwars:
            **kwargs: Optional **kwargs passed to the QueryDb.query() call

        Returns:
            result (pandas.DataFrame or sqlalchemy ResultProxy): Query result
            as a DataFrame (default) or sqlalchemy result.
        """
        col, id_col = self._query_helper(by=None)

        where_string = str(where_string)  # Coerce here, for .__contains___
        where_operators = ["=", ">", "<", "LIKE", "like"]
        if np.any([where_string.__contains__(w) for w in where_operators]):
            select = ("SELECT %s FROM %s WHERE %s" %
                      (col, self.table.name, where_string))
        else:
            select = ("SELECT %s FROM %s WHERE %s = %s" %
                      (col, self.table.name, id_col, where_string))

        return self._db.query(select, **kwargs)