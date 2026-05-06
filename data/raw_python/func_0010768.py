def query(self, sql_query, return_as="dataframe"):
        """
        Execute a raw SQL query against the the SQL DB.

        Args:
            sql_query (str): A raw SQL query to execute.

        Kwargs:
            return_as (str): Specify what type of object should be
            returned. The following are acceptable types:
            - "dataframe": pandas.DataFrame or None if no matching query
            - "result": sqlalchemy.engine.result.ResultProxy

        Returns:
            result (pandas.DataFrame or sqlalchemy ResultProxy): Query result
            as a DataFrame (default) or sqlalchemy result (specified with
            return_as="result")

        Raises:
            QueryDbError
        """
        if isinstance(sql_query, str):
            pass
        elif isinstance(sql_query, unicode):
            sql_query = str(sql_query)
        else:
            raise QueryDbError("query() requires a str or unicode input.")

        query = sqlalchemy.sql.text(sql_query)

        if return_as.upper() in ["DF", "DATAFRAME"]:
            return self._to_df(query, self._engine)
        elif return_as.upper() in ["RESULT", "RESULTPROXY"]:
            with self._engine.connect() as conn:
                result = conn.execute(query)
                return result
        else:
            raise QueryDbError("Other return types not implemented.")