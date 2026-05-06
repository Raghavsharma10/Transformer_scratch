def _as_document(self, partition):
        """ Converts partition to document indexed by to FTS index.

        Args:
            partition (orm.Partition): partition to convert.

        Returns:
            dict with structure matches to BasePartitionIndex._schema.

        """
        doc = super(self.__class__, self)._as_document(partition)

        # SQLite FTS can't find terms with `-`, replace it with underscore here and while searching.
        # See http://stackoverflow.com/questions/3865733/how-do-i-escape-the-character-in-sqlite-fts3-queries
        doc['keywords'] = doc['keywords'].replace('-', '_')
        doc['doc'] = doc['doc'].replace('-', '_')
        doc['title'] = doc['title'].replace('-', '_')

        # pass time_coverage to the _index_document.
        doc['time_coverage'] = partition.time_coverage
        return doc