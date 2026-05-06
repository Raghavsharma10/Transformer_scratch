def insert_many(self, rows, chunk_size=1000):
        """
        Add many rows at a time, which is significantly faster than adding
        them one by one. Per default the rows are processed in chunks of
        1000 per commit, unless you specify a different ``chunk_size``.
        See :py:meth:`insert() <dataset.Table.insert>` for details on
        the other parameters.
        ::
            rows = [dict(name='Dolly')] * 10000
            table.insert_many(rows)
        """

        def _process_chunk(chunk):
            self.table.insert().execute(chunk)

        self._check_dropped()

        chunk = []
        for i, row in enumerate(rows, start=1):
            chunk.append(row)
            if i % chunk_size == 0:
                _process_chunk(chunk)
                chunk = []
        if chunk:
            _process_chunk(chunk)