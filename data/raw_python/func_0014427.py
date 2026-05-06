def index_json(self, collection, data, params=None, min_rf=None, **kwargs):
        """
        :param str collection: The name of the collection for the request.
        :param data str data: Valid Solr JSON as a string. ex: '[{"title": "testing solr indexing", "id": "test1"}]'
        :param min_rf int min_rf: Required number of replicas to write to'

        Sends supplied json to solr for indexing, supplied JSON must be a list of dictionaries.  ::

            >>> docs = [{'id':'changeme','field1':'value1'},
                        {'id':'changeme1','field2':'value2'}]
            >>> solr.index_json('SolrClient_unittest',json.dumps(docs))

        """
        if params is None:
            params = {}

        resp, con_inf = self.transport.send_request(method='POST',
                                                    endpoint='update',
                                                    collection=collection,
                                                    data=data,
                                                    params=params,
                                                    min_rf=min_rf,
                                                    **kwargs)
        if min_rf is not None:
            rf = resp['responseHeader']['rf']
            if rf < min_rf:
                raise MinRfError("couldn't satisfy rf:%s min_rf:%s" % (rf, min_rf), rf=rf, min_rf=min_rf)
        if resp['responseHeader']['status'] == 0:
            return True
        return False