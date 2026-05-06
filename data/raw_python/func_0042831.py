def get_count_sql(self):
        """
        Build a SELECT query which returns the count of items for an unlimited SELECT

        :return:
            A SQL SELECT query which returns the count of items for an unlimited query based on this SQLBuilder
        """
        sql = 'SELECT COUNT(*) FROM ' + self.tables
        if len(self.where_clauses) > 0:
            sql += ' WHERE '
            sql += ' AND '.join(self.where_clauses)
        return sql