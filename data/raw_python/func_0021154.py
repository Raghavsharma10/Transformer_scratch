def describe(self, bucket, descriptor=None):
        """https://github.com/frictionlessdata/tableschema-pandas-py#storage
        """

        # Set descriptor
        if descriptor is not None:
            self.__descriptors[bucket] = descriptor

        # Get descriptor
        else:
            descriptor = self.__descriptors.get(bucket)
            if descriptor is None:
                dataframe = self.__dataframes[bucket]
                descriptor = self.__mapper.restore_descriptor(dataframe)

        return descriptor