def _get_query(self, cursor):
        '''
        Query tempalte for source Solr, sorts by id by default.
        '''
        query = {'q':'*:*',
                'sort':'id desc',
                'rows':self._rows,
                'cursorMark':cursor}
        if self._date_field:
            query['sort'] = "{} asc, id desc".format(self._date_field)
        if self._per_shard:
            query['distrib'] = 'false'
        return query