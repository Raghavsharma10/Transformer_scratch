def iter(self, bucket):
        """https://github.com/frictionlessdata/tableschema-pandas-py#storage
        """

        # Check existense
        if bucket not in self.buckets:
            message = 'Bucket "%s" doesn\'t exist.' % bucket
            raise tableschema.exceptions.StorageError(message)

        # Prepare
        descriptor = self.describe(bucket)
        schema = tableschema.Schema(descriptor)

        # Yield rows
        for pk, row in self.__dataframes[bucket].iterrows():
            row = self.__mapper.restore_row(row, schema=schema, pk=pk)
            yield row