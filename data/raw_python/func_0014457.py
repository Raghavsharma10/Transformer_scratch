def _get_date_range_query(self, start_date, end_date, timespan= 'DAY', date_field= None):
        '''
        Gets counts of items per specified date range.
        :param collection: Solr Collection to use.
        :param timespan: Solr Date Math compliant value for faceting ex HOUR, MONTH, DAY
        '''
        if date_field is None:
            date_field = self._date_field
        query ={'q':'*:*',
                'rows':0,
                'facet':'true',
                'facet.range': date_field,
                'facet.range.gap': '+1{}'.format(timespan),
                'facet.range.end': '{}'.format(end_date),
                'facet.range.start': '{}'.format(start_date),
                'facet.range.include': 'all'
                }
        if self._per_shard:
            query['distrib'] = 'false'
        return query