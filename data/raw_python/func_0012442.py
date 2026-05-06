def select(self, table, columns=None, join=None, where=None, group=None, having=None, order=None, limit=None,
               iterator=False, fetch=True):
        """
        :type table: string
        :type columns: list
        :type join: dict
        :param join: {'[>]table1(t1)': {'user.id': 't1.user_id'}} -> "LEFT JOIN table AS t1 ON user.id = t1.user_id"
        :type where: dict
        :type group: string|list
        :type having: string
        :type order: string|list
        :type limit: int|list
        # TODO: change to offset
        :param limit: The max row number for this query.
                      If it contains offset, limit must be a list like [offset, limit]
        :param iterator: Whether to output the result in a generator. It always returns generator if the cursor is
                         SSCursor or SSDictCursor, no matter iterator is True or False.
        :type fetch: bool
        """
        if not columns:
            columns = ['*']
        where_q, _args = self._where_parser(where)

        # TODO: support multiple table

        _sql = ''.join(['SELECT ', self._backtick_columns(columns),
                        ' FROM ', self._tablename_parser(table)['formatted_tablename'],
                        self._join_parser(join),
                        where_q,
                        (' GROUP BY ' + self._by_columns(group)) if group else '',
                        (' HAVING ' + having) if having else '',
                        (' ORDER BY ' + self._by_columns(order)) if order else '',
                        self._limit_parser(limit), ';'])

        if self.debug:
            return self.cur.mogrify(_sql, _args)

        execute_result = self.cur.execute(_sql, _args)

        if not fetch:
            return execute_result

        if self.cursorclass in (pymysql.cursors.SSCursor, pymysql.cursors.SSDictCursor):
            return self.cur

        if iterator:
            return self._yield_result()

        return self.cur.fetchall()