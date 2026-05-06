def _as_document(self, partition):
        """ Converts partition to document indexed by to FTS index.

        Args:
            partition (orm.Partition): partition to convert.

        Returns:
            dict with structure matches to BasePartitionIndex._schema.

        """
        doc = super(self.__class__, self)._as_document(partition)

        # pass time_coverage to the _index_document.
        doc['time_coverage'] = partition.time_coverage
        return doc