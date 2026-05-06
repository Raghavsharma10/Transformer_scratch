def index(self, collection, docs, params=None, min_rf=None, **kwargs):
        """
        :param str collection: The name of the collection for the request.
        :param docs list docs: List of dicts. ex: [{"title": "testing solr indexing", "id": "test1"}]
        :param min_rf int min_rf: Required number of replicas to write to'

        Sends supplied list of dicts to solr for indexing.  ::

            >>> docs = [{'id':'changeme','field1':'value1'}, {'id':'changeme1','field2':'value2'}]
            >>> solr.index('SolrClient_unittest', docs)

        """
        data = json.dumps(docs)
        return self.index_json(collection, data, params, min_rf=min_rf, **kwargs)