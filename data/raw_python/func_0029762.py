def search(self, search_phrase, limit=None):
        """ Finds datasets by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to return. None means without limit.

        Returns:
            list of DatasetSearchResult instances.

        """

        query, query_params = self._make_query_from_terms(search_phrase, limit=limit)

        self._parsed_query = (str(query), query_params)

        assert isinstance(query, TextClause)

        datasets = {}

        def make_result(vid=None, b_score=0, p_score=0):
            res = DatasetSearchResult()
            res.b_score = b_score
            res.p_score = p_score
            res.partitions = set()
            res.vid = vid
            return res

        if query_params:
            results = self.execute(query, **query_params)

            for result in results:
                vid, dataset_score = result

                datasets[vid] = make_result(vid, b_score=dataset_score)


        logger.debug('Extending datasets with partitions.')

        for partition in self.backend.partition_index.search(search_phrase):

            if partition.dataset_vid not in datasets:
                datasets[partition.dataset_vid] = make_result(partition.dataset_vid)

            datasets[partition.dataset_vid].p_score += partition.score
            datasets[partition.dataset_vid].partitions.add(partition)

        return list(datasets.values())