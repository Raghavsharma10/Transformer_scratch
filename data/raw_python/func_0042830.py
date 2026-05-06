def get_select_sql(self, columns, order=None, limit=0, skip=0):
        """
        Build a SELECT query based on the current state of the builder.

        :param columns:
            SQL fragment describing which columns to select i.e. 'e.obstoryID, s.statusID'
        :param order:
            Optional ordering constraint, i.e. 'e.eventTime DESC'
        :param limit:
            Optional, used to build the 'LIMIT n' clause. If not specified no limit is imposed.
        :param skip:
            Optional, used to build the 'OFFSET n' clause. If not specified results are returned from the first item
            available. Note that this parameter must be combined with 'order', otherwise there's no ordering imposed
            on the results and subsequent queries may return overlapping data randomly. It's unlikely that this will
            actually happen as almost all databases do in fact create an internal ordering, but there's no guarantee
            of this (and some operations such as indexing will definitely break this property unless explicitly set).
        :returns:
            A SQL SELECT query, which will make use of self.sql_args when executed.
        """
        sql = 'SELECT '
        sql += '{0} FROM {1} '.format(columns, self.tables)
        if len(self.where_clauses) > 0:
            sql += ' WHERE '
            sql += ' AND '.join(self.where_clauses)
        if order is not None:
            sql += ' ORDER BY {0}'.format(order)
        if limit > 0:
            sql += ' LIMIT {0} '.format(limit)
        if skip > 0:
            sql += ' OFFSET {0} '.format(skip)
        return sql