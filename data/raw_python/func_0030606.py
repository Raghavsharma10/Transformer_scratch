def index(self, connection, partition, columns):
        """ Create an index on the columns.

        Args:
            connection:
            partition (orm.Partition):
            columns (list of str):

        """
        query_tmpl = 'CREATE INDEX ON {table_name} ({column});'
        table_name = '{}_v'.format(partition.vid)
        for column in columns:
            query = query_tmpl.format(table_name=table_name, column=column)
            logger.debug('Creating postgres index.\n    column: {}, query: {}'.format(column, query))
            with connection.cursor() as cursor:
                cursor.execute(query)
                cursor.execute('COMMIT;')