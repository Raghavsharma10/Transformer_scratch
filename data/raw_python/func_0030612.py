def _relation_exists(cls, connection, relation):
        """ Returns True if relation exists in the postgres db. Otherwise returns False.

        Args:
            connection: connection to postgres database who stores mpr data.
            relation (str): name of the table, view or materialized view.

        Note:
            relation means table, view or materialized view here.

        Returns:
            boolean: True if relation exists, False otherwise.

        """
        schema_name, table_name = relation.split('.')

        exists_query = '''
            SELECT 1
            FROM   pg_catalog.pg_class c
            JOIN   pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE  n.nspname = %s
            AND    c.relname = %s
            AND    (c.relkind = 'r' OR c.relkind = 'v' OR c.relkind = 'm')
                -- r - table, v - view, m - materialized view.
        '''
        with connection.cursor() as cursor:
            cursor.execute(exists_query, [schema_name, table_name])
            result = cursor.fetchall()
            return result == [(1,)]