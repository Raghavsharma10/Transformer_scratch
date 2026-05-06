def _as_document(self, dataset):
        """ Converts dataset to document indexed by to FTS index.

        Args:
            dataset (orm.Dataset): dataset to convert.

        Returns:
            dict with structure matches to BaseDatasetIndex._schema.

        """
        assert isinstance(dataset, Dataset)

        doc = super(self.__class__, self)._as_document(dataset)

        # SQLite FTS can't find terms with `-`, replace it with underscore here and while searching.
        # See http://stackoverflow.com/questions/3865733/how-do-i-escape-the-character-in-sqlite-fts3-queries
        doc['keywords'] = doc['keywords'].replace('-', '_')
        doc['doc'] = doc['doc'].replace('-', '_')
        doc['title'] = doc['title'].replace('-', '_')
        return doc