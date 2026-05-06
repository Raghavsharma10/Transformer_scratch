def list_benchmarks(self,
            index    = None,
            doc_type = None,
            params   = {},
            cb       = None,
            **kwargs
        ):
        """
        View the progress of long-running benchmarks.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/master/search-benchmark.html>`_
        :arg index: A comma-separated list of index names; use `_all` or empty
            string to perform the operation on all indices
        :arg doc_type: The name of the document type
        """

        url = self.mk_url(*[index, doc_type, '_bench'], **params)

        self.client.fetch(
            self.mk_req(url, method='GET', **kwargs),
            callback = callback
        )