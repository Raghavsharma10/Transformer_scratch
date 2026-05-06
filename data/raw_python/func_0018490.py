def dataframe(self, table):
        """
        create a pandas dataframe from a table or query

        Parameters
        ----------

        table : table
            a table in this database or a query

        limit: integer
            an integer limit on the query

        offset: integer
            an offset for the query
        """
        from pandas import DataFrame
        if isinstance(table, six.string_types):
            table = getattr(self, table)
        try:
            rec = table.first()
        except AttributeError:
            rec = table[0]
        if hasattr(table, "all"):
            records = table.all()
        else:
            records = [tuple(t) for t in table]
        cols = [c.name for c in rec._table.columns]
        return DataFrame.from_records(records, columns=cols)