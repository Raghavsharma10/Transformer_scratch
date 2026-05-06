def search(self, search_phrase, limit=None):
        """ Finds partitions by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to generate. None means without limit.

        Generates:
            PartitionSearchResult instances.
        """
        query, query_params = self._make_query_from_terms(search_phrase, limit=limit)

        self._parsed_query = (str(query), query_params)

        if query is not None:

            self.backend.library.database.set_connection_search_path()

            results = self.execute(query, **query_params)

            for result in results:
                vid, dataset_vid, score = result
                yield PartitionSearchResult(
                    vid=vid, dataset_vid=dataset_vid, score=score)