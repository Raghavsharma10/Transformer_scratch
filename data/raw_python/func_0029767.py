def _make_query_from_terms(self, terms, limit=None):
        """ Creates a query for partition from decomposed search terms.

        Args:
            terms (dict or unicode or string):

        Returns:
            tuple of (TextClause, dict): First element is FTS query, second is
            parameters of the query. Element of the execution of the query is
            tuple of three elements: (vid, dataset_vid, score).

        """
        expanded_terms = self._expand_terms(terms)
        terms_used = 0

        if expanded_terms['doc']:
            # create query with real score.
            query_parts = ["SELECT vid, dataset_vid, ts_rank_cd(setweight(doc,'C'), to_tsquery(:doc)) as score"]
        if expanded_terms['doc'] and expanded_terms['keywords']:
            query_parts = ["SELECT vid, dataset_vid, ts_rank_cd(setweight(doc,'C'), to_tsquery(:doc)) "
                           " +  ts_rank_cd(setweight(to_tsvector(coalesce(keywords::text,'')),'B'), to_tsquery(:keywords))"
                           ' as score']
        else:
            # create query with score = 1 because query will not touch doc field.
            query_parts = ['SELECT vid, dataset_vid, 1 as score']

        query_parts.append('FROM partition_index')
        query_params = {}
        where_count = 0

        if expanded_terms['doc']:
            query_parts.append('WHERE doc @@ to_tsquery(:doc)')
            query_params['doc'] = self.backend._and_join(expanded_terms['doc'])
            where_count += 1
            terms_used += 1

        if expanded_terms['keywords']:
            query_params['keywords'] = self.backend._and_join(expanded_terms['keywords'])

            kw_q = "to_tsvector(coalesce(keywords::text,'')) @@ to_tsquery(:keywords)"

            query_parts.append(("AND " if where_count else "WHERE ") + kw_q)

            where_count += 1
            terms_used += 1

        if expanded_terms['from']:

            query_parts.append(("AND " if where_count else "WHERE ") + ' from_year >= :from_year')

            query_params['from_year'] = expanded_terms['from']
            where_count += 1
            terms_used += 1

        if expanded_terms['to']:

            query_parts.append(("AND " if where_count else "WHERE ") + ' to_year <= :to_year')

            query_params['to_year'] = expanded_terms['to']
            where_count += 1
            terms_used += 1

        query_parts.append('ORDER BY score DESC')

        if limit:
            query_parts.append('LIMIT :limit')
            query_params['limit'] = limit

        if not terms_used:
            logger.debug('No terms used; not creating query')
            return None, None

        query_parts.append(';')
        deb_msg = 'Dataset terms conversion: `{}` terms converted to `{}` with `{}` params query.'\
            .format(terms, query_parts, query_params)
        logger.debug(deb_msg)

        return text('\n'.join(query_parts)), query_params