def _make_query_from_terms(self, terms, limit=None):
        """ Creates a query for dataset from decomposed search terms.

        Args:
            terms (dict or unicode or string):

        Returns:
            tuple of (TextClause, dict): First element is FTS query, second is parameters
                of the query. Element of the execution of the query is pair: (vid, score).

        """

        expanded_terms = self._expand_terms(terms)

        if expanded_terms['doc']:
            # create query with real score.
            query_parts = ["SELECT vid, ts_rank_cd(setweight(doc,'C'), to_tsquery(:doc)) as score"]
        if expanded_terms['doc'] and expanded_terms['keywords']:
            query_parts = ["SELECT vid, ts_rank_cd(setweight(doc,'C'), to_tsquery(:doc)) "
                           " +  ts_rank_cd(setweight(to_tsvector(coalesce(keywords::text,'')),'B'), to_tsquery(:keywords))"
                           ' as score']
        else:
            # create query with score = 1 because query will not touch doc field.
            query_parts = ['SELECT vid, 1 as score']

        query_parts.append('FROM dataset_index')
        query_params = {}
        where_counter = 0

        if expanded_terms['doc']:
            where_counter += 1
            query_parts.append('WHERE doc @@ to_tsquery(:doc)')
            query_params['doc'] = self.backend._and_join(expanded_terms['doc'])

        if expanded_terms['keywords']:

            query_params['keywords'] = self.backend._and_join(expanded_terms['keywords'])

            kw_q = "to_tsvector(coalesce(keywords::text,'')) @@ to_tsquery(:keywords)"

            query_parts.append( ("AND " if where_counter else "WHERE ") + kw_q )


        query_parts.append('ORDER BY score DESC')
        if limit:
            query_parts.append('LIMIT :limit')
            query_params['limit'] = limit

        query_parts.append(';')
        deb_msg = 'Dataset terms conversion: `{}` terms converted to `{}` with `{}` params query.'\
            .format(terms, query_parts, query_params)
        logger.debug(deb_msg)


        q = text('\n'.join(query_parts)), query_params
        logger.debug('Dataset search query: {}'.format(q))
        return q