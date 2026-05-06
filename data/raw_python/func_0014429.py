def mget(self, collection, doc_ids, **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param tuple doc_ids: ID of the document to be retrieved.

        Retrieve documents from Solr based on the ID. ::

            >>> solr.get('SolrClient_unittest','changeme')
        """

        resp, con_inf = self.transport.send_request(method='GET',
                                                    endpoint='get',
                                                    collection=collection,
                                                    params={'ids': doc_ids},
                                                    **kwargs)
        if 'docs' in resp['response']:
            return resp['response']['docs']
        raise NotFoundError