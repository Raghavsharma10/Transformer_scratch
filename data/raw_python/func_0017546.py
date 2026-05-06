def search(self, resource, resource_class, search_filter=None, dmql_query=None, limit=9999999, offset=0,
               optional_parameters=None, auto_offset=True, query_type='DMQL2', standard_names=0,
               response_format='COMPACT-DECODED'):
        """
        Preform a search on the RETS board
        :param resource: The resource that contains the class to search
        :param resource_class: The class to search
        :param search_filter: The query as a dict
        :param dmql_query: The query in dmql format
        :param limit: Limit search values count
        :param offset: Offset for RETS request. Useful when RETS limits number of results or transactions
        :param optional_parameters: Values for option paramters
        :param auto_offset: Should the search be allowed to trigger subsequent searches.
        :param query_type: DMQL or DMQL2 depending on the rets server.
        :param standard_names: 1 to use standard names, 0 to use system names
        :param response_format: COMPACT-DECODED, COMPACT, or STANDARD-XML
        :return: dict
        """

        if (search_filter and dmql_query) or (not search_filter and not dmql_query):
            raise ValueError("You may specify either a search_filter or dmql_query")

        search_helper = DMQLHelper()

        if dmql_query:
            dmql_query = search_helper.dmql(query=dmql_query)
        else:
            dmql_query = search_helper.filter_to_dmql(filter_dict=search_filter)

        parameters = {
            'SearchType': resource,
            'Class': resource_class,
            'Query': dmql_query,
            'QueryType': query_type,
            'Count': 1,
            'Format': response_format,
            'StandardNames': standard_names,
        }

        if not optional_parameters:
            optional_parameters = {}
        parameters.update(optional_parameters)

        # if the Select parameter given is an array, format it as it needs to be
        if 'Select' in parameters and isinstance(parameters.get('Select'), list):
            parameters['Select'] = ','.join(parameters['Select'])

        if limit:
            parameters['Limit'] = limit

        if offset:
            parameters['Offset'] = offset

        search_cursor = OneXSearchCursor()
        response = self._request(
            capability='Search',
            options={
                'query': parameters,
            },
            stream=True
        )
        try:
            return search_cursor.generator(response=response)

        except MaxrowException as max_exception:
            # Recursive searching if automatically performing offsets for the  client
            if auto_offset and limit > len(max_exception.rows_returned):
                new_limit = limit - len(max_exception.rows_returned)  # have not returned results to the desired limit
                new_offset = offset + len(max_exception.rows_returned)  # adjust offset
                results = self.search(resource=resource, resource_class=resource_class, search_filter=None,
                                      dmql_query=dmql_query, offset=new_offset, limit=new_limit,
                                      optional_parameters=optional_parameters, auto_offset=auto_offset)

                previous_results = max_exception.rows_returned
                return previous_results + results
            return max_exception.rows_returned