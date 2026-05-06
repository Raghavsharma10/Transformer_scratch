def search(self, search_phrase, limit=None):
        """ Finds partitions by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to generate. None means without limit.

        Yields:
            PartitionSearchResult instances.

        """

        query_string = self._make_query_from_terms(search_phrase)
        self._parsed_query = query_string
        schema = self._get_generic_schema()
        parser = QueryParser('doc', schema=schema)
        query = parser.parse(query_string)
        logger.debug('Searching partitions using `{}` query.'.format(query))
        with self.index.searcher() as searcher:
            results = searcher.search(query, limit=limit)
            for hit in results:
                yield PartitionSearchResult(
                    vid=hit['vid'], dataset_vid=hit['dataset_vid'], score=hit.score)