def distinct(self, *columns, **_filter):
        """
        Returns all rows of a table, but removes rows in with duplicate values in ``columns``.
        Interally this creates a `DISTINCT statement <http://www.w3schools.com/sql/sql_distinct.asp>`_.
        ::

            # returns only one row per year, ignoring the rest
            table.distinct('year')
            # works with multiple columns, too
            table.distinct('year', 'country')
            # you can also combine this with a filter
            table.distinct('year', country='China')
        """
        self._check_dropped()
        qargs = []
        try:
            columns = [self.table.c[c] for c in columns]
            for col, val in _filter.items():
                qargs.append(self.table.c[col] == val)
        except KeyError:
            return []

        q = expression.select(
            columns,
            distinct=True,
            whereclause=and_(*qargs),
            order_by=[c.asc() for c in columns],
        )
        # if just looking at one column, return a simple list
        if len(columns) == 1:
            return itertools.chain.from_iterable(self.engine.execute(q))
        # otherwise return specified row_type
        else:
            return ResultIter(self.engine.execute(q), row_type=self.db.row_type)