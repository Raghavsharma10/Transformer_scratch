def delete_doc_by_id(self, collection, doc_id, **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param str id: ID of the document to be deleted. Can specify '*' to delete everything.

        Deletes items from Solr based on the ID. ::

            >>> solr.delete_doc_by_id('SolrClient_unittest','changeme')

        """
        if ' ' in doc_id:
            doc_id = '"{}"'.format(doc_id)
        temp = {"delete": {"query": 'id:{}'.format(doc_id)}}
        resp, con_inf = self.transport.send_request(method='POST',
                                                    endpoint='update',
                                                    collection=collection,
                                                    data=json.dumps(temp),
                                                    **kwargs)
        return resp