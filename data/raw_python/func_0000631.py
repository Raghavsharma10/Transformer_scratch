def create(self, bucket, descriptor, force=False, indexes_fields=None):
        """https://github.com/frictionlessdata/tableschema-sql-py#storage
        """

        # Make lists
        buckets = bucket
        if isinstance(bucket, six.string_types):
            buckets = [bucket]
        descriptors = descriptor
        if isinstance(descriptor, dict):
            descriptors = [descriptor]
        if indexes_fields is None or len(indexes_fields) == 0:
            indexes_fields = [()] * len(descriptors)
        elif type(indexes_fields[0][0]) not in {list, tuple}:
            indexes_fields = [indexes_fields]

        # Check dimensions
        if not (len(buckets) == len(descriptors) == len(indexes_fields)):
            raise tableschema.exceptions.StorageError('Wrong argument dimensions')

        # Check buckets for existence
        for bucket in reversed(self.buckets):
            if bucket in buckets:
                if not force:
                    message = 'Bucket "%s" already exists.' % bucket
                    raise tableschema.exceptions.StorageError(message)
                self.delete(bucket)

        # Define buckets
        for bucket, descriptor, index_fields in zip(buckets, descriptors, indexes_fields):
            tableschema.validate(descriptor)
            table_name = self.__mapper.convert_bucket(bucket)
            columns, constraints, indexes, fallbacks, table_comment = self.__mapper \
                .convert_descriptor(bucket, descriptor, index_fields, self.__autoincrement)
            Table(table_name, self.__metadata, *(columns + constraints + indexes),
                  comment=table_comment)
            self.__descriptors[bucket] = descriptor
            self.__fallbacks[bucket] = fallbacks

        # Create tables, update metadata
        self.__metadata.create_all()