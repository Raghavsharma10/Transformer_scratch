def delete_doc_by_query(self, collection, query, **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param str query: Query selecting documents to be deleted.

        Deletes items from Solr based on a given query. ::

            >>> solr.delete_doc_by_query('SolrClient_unittest','*:*')

        """
        temp = {"delete": {"query": query}}
        resp, con_inf = self.transport.send_request(method='POST',
                                                    endpoint='update',
                                                    collection=collection,
                                                    data=json.dumps(temp),
                                                    **kwargs)
        return resp