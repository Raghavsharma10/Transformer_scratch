def _sort2sql(self, sort):
        """
        RETURN ORDER BY CLAUSE
        """
        if not sort:
            return ""
        return SQL_ORDERBY + sql_list([quote_column(o.field) + (" DESC" if o.sort == -1 else "") for o in sort])