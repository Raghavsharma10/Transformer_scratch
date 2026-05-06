def create(self, bucket, descriptor, force=False):
        """https://github.com/frictionlessdata/tableschema-pandas-py#storage
        """

        # Make lists
        buckets = bucket
        if isinstance(bucket, six.string_types):
            buckets = [bucket]
        descriptors = descriptor
        if isinstance(descriptor, dict):
            descriptors = [descriptor]

        # Check buckets for existence
        for bucket in buckets:
            if bucket in self.buckets:
                if not force:
                    message = 'Bucket "%s" already exists' % bucket
                    raise tableschema.exceptions.StorageError(message)
                self.delete(bucket)

        # Define dataframes
        for bucket, descriptor in zip(buckets, descriptors):
            tableschema.validate(descriptor)
            self.__descriptors[bucket] = descriptor
            self.__dataframes[bucket] = pd.DataFrame()