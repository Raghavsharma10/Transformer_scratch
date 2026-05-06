def iter(self, bucket):
        """https://github.com/frictionlessdata/tableschema-sql-py#storage
        """

        # Get table and fallbacks
        table = self.__get_table(bucket)
        schema = tableschema.Schema(self.describe(bucket))

        # Open and close transaction
        with self.__connection.begin():
            # Streaming could be not working for some backends:
            # http://docs.sqlalchemy.org/en/latest/core/connections.html
            select = table.select().execution_options(stream_results=True)
            result = select.execute()
            for row in result:
                row = self.__mapper.restore_row(row, schema=schema)
                yield row