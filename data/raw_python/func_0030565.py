def search(self, search_phrase, limit=None):
        """ Finds datasets by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to return. None means without limit.

        Returns:
            list of DatasetSearchResult instances.

        """
        query_string = self._make_query_from_terms(search_phrase)
        self._parsed_query = query_string
        schema = self._get_generic_schema()

        parser = QueryParser('doc', schema=schema)

        query = parser.parse(query_string)

        datasets = defaultdict(DatasetSearchResult)

        # collect all datasets
        logger.debug('Searching datasets using `{}` query.'.format(query))
        with self.index.searcher() as searcher:
            results = searcher.search(query, limit=limit)
            for hit in results:
                vid = hit['vid']
                datasets[vid].vid = hit['vid']
                datasets[vid].b_score += hit.score

        # extend datasets with partitions
        logger.debug('Extending datasets with partitions.')
        for partition in self.backend.partition_index.search(search_phrase):
            datasets[partition.dataset_vid].p_score += partition.score
            datasets[partition.dataset_vid].partitions.add(partition)
        return list(datasets.values())