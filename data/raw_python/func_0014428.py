def get(self, collection, doc_id, **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param str doc_id: ID of the document to be retrieved.

        Retrieve document from Solr based on the ID. ::

            >>> solr.get('SolrClient_unittest','changeme')
        """

        resp, con_inf = self.transport.send_request(method='GET',
                                                    endpoint='get',
                                                    collection=collection,
                                                    params={'id': doc_id},
                                                    **kwargs)
        if 'doc' in resp and resp['doc']:
            return resp['doc']
        raise NotFoundError