def search(self, search_phrase, limit=None):
        """ Finds partitions by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to generate. None means without limit.

        Generates:
            PartitionSearchResult instances.
        """

        # SQLite FTS can't find terms with `-`, therefore all hyphens replaced with underscore before save.
        # Now to make proper query we need to replace all hyphens in the search phrase.
        # See http://stackoverflow.com/questions/3865733/how-do-i-escape-the-character-in-sqlite-fts3-queries
        search_phrase = search_phrase.replace('-', '_')
        terms = SearchTermParser().parse(search_phrase)
        from_year = terms.pop('from', None)
        to_year = terms.pop('to', None)

        query, query_params = self._make_query_from_terms(terms)

        self._parsed_query = (query, query_params)

        connection = self.backend.library.database.connection

        connection.connection.create_function('rank', 1, _make_rank_func((1., .1, 0, 0)))

        # SQLite FTS implementation does not allow to create indexes on FTS tables.
        # see https://sqlite.org/fts3.html 1.5. Summary, p 1:
        # ... it is not possible to create indices ...
        #
        # So, filter years range here.

        results = connection.execute(query, query_params).fetchall()

        for result in results:
            vid, dataset_vid, score, db_from_year, db_to_year = result
            if from_year and from_year < db_from_year:
                continue
            if to_year and to_year > db_to_year:
                continue
            yield PartitionSearchResult(
                vid=vid, dataset_vid=dataset_vid, score=score)