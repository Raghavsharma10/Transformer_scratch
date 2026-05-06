def write(self, bucket, rows, keyed=False, as_generator=False, update_keys=None):
        """https://github.com/frictionlessdata/tableschema-sql-py#storage
        """

        # Check update keys
        if update_keys is not None and len(update_keys) == 0:
            message = 'Argument "update_keys" cannot be an empty list'
            raise tableschema.exceptions.StorageError(message)

        # Get table and description
        table = self.__get_table(bucket)
        schema = tableschema.Schema(self.describe(bucket))
        fallbacks = self.__fallbacks.get(bucket, [])

        # Write rows to table
        convert_row = partial(self.__mapper.convert_row, schema=schema, fallbacks=fallbacks)
        writer = Writer(table, schema, update_keys, self.__autoincrement, convert_row)
        with self.__connection.begin():
            gen = writer.write(rows, keyed=keyed)
            if as_generator:
                return gen
            collections.deque(gen, maxlen=0)