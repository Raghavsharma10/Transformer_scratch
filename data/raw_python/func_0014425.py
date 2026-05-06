def query(self, collection, query, request_handler='select', **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param str request_handler: Request handler, default is 'select'
        :param dict query: Python dictonary of Solr query parameters.

        Sends a query to Solr, returns a SolrResults Object. `query` should be a dictionary of solr request handler arguments.
        Example::

            res = solr.query('SolrClient_unittest',{
                'q':'*:*',
                'facet':True,
                'facet.field':'facet_test',
            })

        """
        for field in ['facet.pivot']:
            if field in query.keys():
                if type(query[field]) is str:
                    query[field] = query[field].replace(' ', '')
                elif type(query[field]) is list:
                    query[field] = [s.replace(' ', '') for s in query[field]]

        method = 'POST'
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        params = query
        data = {}
        resp, con_inf = self.transport.send_request(method=method,
                                                    endpoint=request_handler,
                                                    collection=collection,
                                                    params=params,
                                                    data=data,
                                                    headers=headers,
                                                    **kwargs)
        if resp:
            resp = SolrResponse(resp)
            resp.url = con_inf['url']
            return resp