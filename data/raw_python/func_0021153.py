def delete(self, bucket=None, ignore=False):
        """https://github.com/frictionlessdata/tableschema-pandas-py#storage
        """

        # Make lists
        buckets = bucket
        if isinstance(bucket, six.string_types):
            buckets = [bucket]
        elif bucket is None:
            buckets = reversed(self.buckets)

        # Iterate over buckets
        for bucket in buckets:

            # Non existent bucket
            if bucket not in self.buckets:
                if not ignore:
                    message = 'Bucket "%s" doesn\'t exist' % bucket
                    raise tableschema.exceptions.StorageError(message)
                return

            # Remove from descriptors
            if bucket in self.__descriptors:
                del self.__descriptors[bucket]

            # Remove from dataframes
            if bucket in self.__dataframes:
                del self.__dataframes[bucket]