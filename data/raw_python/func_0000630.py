def buckets(self):
        """https://github.com/frictionlessdata/tableschema-sql-py#storage
        """
        buckets = []
        for table in self.__metadata.sorted_tables:
            bucket = self.__mapper.restore_bucket(table.name)
            if bucket is not None:
                buckets.append(bucket)
        return buckets