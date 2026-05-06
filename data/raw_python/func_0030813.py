def search(self, search_phrase, limit=None):
        """ Finds datasets by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to return. None means without limit.

        Returns:
            list of DatasetSearchResult instances.

        """
        # SQLite FTS can't find terms with `-`, therefore all hyphens were replaced with underscore
        # before save. Now to get appropriate result we need to replace all hyphens in the search phrase.
        # See http://stackoverflow.com/questions/3865733/how-do-i-escape-the-character-in-sqlite-fts3-queries
        search_phrase = search_phrase.replace('-', '_')
        query, query_params = self._make_query_from_terms(search_phrase)

        self._parsed_query = (query, query_params)

        connection = self.backend.library.database.connection
        # Operate on the raw connection
        connection.connection.create_function('rank', 1, _make_rank_func((1., .1, 0, 0)))

        logger.debug('Searching datasets using `{}` query.'.format(query))
        results = connection.execute(query,
                                     **query_params).fetchall()  # Query on the Sqlite proxy to the raw connection

        datasets = defaultdict(DatasetSearchResult)
        for result in results:
            vid, score = result
            datasets[vid] = DatasetSearchResult()
            datasets[vid].vid = vid
            datasets[vid].b_score = score

        logger.debug('Extending datasets with partitions.')
        for partition in self.backend.partition_index.search(search_phrase):
            datasets[partition.dataset_vid].p_score += partition.score
            datasets[partition.dataset_vid].partitions.add(partition)
        return list(datasets.values())