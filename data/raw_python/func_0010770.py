def _to_df(self, query, conn, index_col=None, coerce_float=True, params=None,
               parse_dates=None, columns=None):
        """
        Internal convert-to-DataFrame convenience wrapper.
        """
        return pd.io.sql.read_sql(str(query), conn, index_col=index_col,
                                  coerce_float=coerce_float, params=params,
                                  parse_dates=parse_dates, columns=columns)