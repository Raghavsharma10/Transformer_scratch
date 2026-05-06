def index(self, connection, partition, columns):
        """ Create an index on the columns.

        Args:
            connection (apsw.Connection): connection to sqlite database who stores mpr table or view.
            partition (orm.Partition):
            columns (list of str):
        """

        import hashlib

        query_tmpl = '''
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns});
        '''

        if not isinstance(columns,(list,tuple)):
            columns = [columns]

        col_list = ','.join('"{}"'.format(col) for col in columns)

        col_hash = hashlib.md5(col_list).hexdigest()

        try:
            table_name = partition.vid
        except AttributeError:
            table_name = partition # Its really a table name

        query = query_tmpl.format(
            index_name='{}_{}_i'.format(table_name, col_hash), table_name=table_name,
            columns=col_list)

        logger.debug('Creating sqlite index: query: {}'.format(query))
        cursor = connection.cursor()

        cursor.execute(query)