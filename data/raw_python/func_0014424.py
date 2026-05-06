def query_raw(self, collection, query, request_handler='select', **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param str request_handler: Request handler, default is 'select'
        :param dict query: Python dictionary of Solr query parameters.

        Sends a query to Solr, returns a dict. `query` should be a dictionary of solr request handler arguments.
        Example::

            res = solr.query_raw('SolrClient_unittest',{
                'q':'*:*',
                'facet':True,
                'facet.field':'facet_test',
            })

        """
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        data = query
        resp, con_inf = self.transport.send_request(method='POST',
                                                    endpoint=request_handler,
                                                    collection=collection,
                                                    data=data,
                                                    headers=headers,
                                                    **kwargs)
        return resp