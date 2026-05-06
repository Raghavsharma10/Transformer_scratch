def describe(self, bucket, descriptor=None):
        """https://github.com/frictionlessdata/tableschema-sql-py#storage
        """

        # Set descriptor
        if descriptor is not None:
            self.__descriptors[bucket] = descriptor

        # Get descriptor
        else:
            descriptor = self.__descriptors.get(bucket)
            if descriptor is None:
                table = self.__get_table(bucket)
                descriptor = self.__mapper.restore_descriptor(
                    table.name, table.columns, table.constraints, self.__autoincrement)

        return descriptor