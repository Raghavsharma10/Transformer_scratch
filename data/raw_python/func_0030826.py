def _make_query_from_terms(self, terms):
        """ Creates a query for partition from decomposed search terms.

        Args:
            terms (dict or unicode or string):

        Returns:
            tuple of (str, dict): First element is str with FTS query, second is parameters of the query.

        """

        match_query = ''

        expanded_terms = self._expand_terms(terms)
        if expanded_terms['doc']:
            match_query = self.backend._and_join(expanded_terms['doc'])

        if expanded_terms['keywords']:
            if match_query:
                match_query = self.backend._and_join(
                    [match_query, self.backend._join_keywords(expanded_terms['keywords'])])
            else:
                match_query = self.backend._join_keywords(expanded_terms['keywords'])

        if match_query:
            query = text("""
                SELECT vid, dataset_vid, rank(matchinfo(partition_index)) AS score, from_year, to_year
                FROM partition_index
                WHERE partition_index MATCH :match_query
                ORDER BY score DESC;
            """)
            query_params = {
                'match_query': match_query}
        else:
            query = text("""
                SELECT vid, dataset_vid, rank(matchinfo(partition_index)), from_year, to_year AS score
                FROM partition_index""")
            query_params = {}

        return query, query_params