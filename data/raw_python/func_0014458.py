def _get_edge_date(self, date_field, sort):
        '''
        This method is used to get start and end dates for the collection.
        '''
        return self._source.query(self._source_coll, {
                'q':'*:*',
                'rows':1,
                'fq':'+{}:*'.format(date_field),
                'sort':'{} {}'.format(date_field, sort)}).docs[0][date_field]