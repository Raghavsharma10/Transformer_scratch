def build_query(self, sql, lookup):
        """
        Modify table and field name variables in a sql string with a dict.
        This seems to be discouraged by psycopg2 docs but it makes small
        adjustments to large sql strings much easier, making prepped queries
        much more versatile.

        USAGE
        sql = 'SELECT $myInputField FROM $myInputTable'
        lookup = {'myInputField':'customer_id', 'myInputTable':'customers'}
        sql = db.build_query(sql, lookup)

        """
        for key, val in six.iteritems(lookup):
            sql = sql.replace("$" + key, val)
        return sql