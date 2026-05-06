def _split_sql(self, sql):
        """
        Generate hunks of SQL that are between the bookends.
        note: beginning & end of string are returned as None

        :return: tuple of beginning bookend, closing bookend, and contents
        """
        bookends = ("\n", ";", "--", "/*", "*/")
        last_bookend_found = None
        start = 0

        while start <= len(sql):
            results = self._get_next_occurrence(sql, start, bookends)
            if results is None:
                yield (last_bookend_found, None, sql[start:])
                start = len(sql) + 1
            else:
                (end, bookend) = results
                yield (last_bookend_found, bookend, sql[start:end])
                start = end + len(bookend)
                last_bookend_found = bookend