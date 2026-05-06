def aggregate(self, **params):
        """ Perform aggreration

        Arguments:
            :_aggregations_params: Dict of aggregation params. Root key is an
                aggregation name. Required.
            :_raise_on_empty: Boolean indicating whether to raise exception
                when IndexNotFoundException exception happens. Optional,
                defaults to False.
        """
        _aggregations_params = params.pop('_aggregations_params', None)
        _raise_on_empty = params.pop('_raise_on_empty', False)

        if not _aggregations_params:
            raise Exception('Missing _aggregations_params')

        # Set limit so ES won't complain. It is ignored in the end
        params['_limit'] = 0
        search_params = self.build_search_params(params)
        search_params.pop('size', None)
        search_params.pop('from_', None)
        search_params.pop('sort', None)

        search_params['body']['aggregations'] = _aggregations_params

        log.debug('Performing aggregation: {}'.format(_aggregations_params))
        try:
            response = self.api.search(**search_params)
        except IndexNotFoundException:
            if _raise_on_empty:
                raise JHTTPNotFound(
                    'Aggregation failed: Index does not exist')
            return {}

        try:
            return response['aggregations']
        except KeyError:
            raise JHTTPNotFound('No aggregations returned from ES')