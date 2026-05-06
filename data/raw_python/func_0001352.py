def find(
        self,
        _limit=None,
        _offset=0,
        _step=5000,
        order_by="id",
        return_count=False,
        **_filter
    ):
        """
        Performs a simple search on the table. Simply pass keyword arguments as ``filter``.
        ::
            results = table.find(country='France')
            results = table.find(country='France', year=1980)
        Using ``_limit``::
            # just return the first 10 rows
            results = table.find(country='France', _limit=10)
        You can sort the results by single or multiple columns. Append a minus sign
        to the column name for descending order::
            # sort results by a column 'year'
            results = table.find(country='France', order_by='year')
            # return all rows sorted by multiple columns (by year in descending order)
            results = table.find(order_by=['country', '-year'])
        By default :py:meth:`find() <dataset.Table.find>` will break the
        query into chunks of ``_step`` rows to prevent huge tables
        from being loaded into memory at once.
        For more complex queries, please use :py:meth:`db.query()`
        instead."""
        self._check_dropped()
        if not isinstance(order_by, (list, tuple)):
            order_by = [order_by]
        order_by = [
            o
            for o in order_by
            if (o.startswith("-") and o[1:] or o) in self.table.columns
        ]
        order_by = [self._args_to_order_by(o) for o in order_by]

        args = self._args_to_clause(_filter)

        # query total number of rows first
        count_query = alias(
            self.table.select(whereclause=args, limit=_limit, offset=_offset),
            name="count_query_alias",
        ).count()
        rp = self.engine.execute(count_query)
        total_row_count = rp.fetchone()[0]
        if return_count:
            return total_row_count

        if _limit is None:
            _limit = total_row_count

        if _step is None or _step is False or _step == 0:
            _step = total_row_count

        if total_row_count > _step and not order_by:
            _step = total_row_count
            log.warn(
                "query cannot be broken into smaller sections because it is unordered"
            )

        queries = []

        for i in count():
            qoffset = _offset + (_step * i)
            qlimit = min(_limit - (_step * i), _step)
            if qlimit <= 0:
                break
            queries.append(
                self.table.select(
                    whereclause=args, limit=qlimit, offset=qoffset, order_by=order_by
                )
            )
        return ResultIter(
            (self.engine.execute(q) for q in queries), row_type=self.db.row_type
        )