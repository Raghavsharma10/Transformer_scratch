def abort_benchmark(self, name=None, params={}, body='', callback=None, **kwargs):
        """
        Aborts a running benchmark.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/master/search-benchmark.html>`_
        :arg name: A benchmark name
        """

        url = self.mk_url(*['_bench', 'abort', name])

        self.client.fetch(
            self.mk_req(url, method='POST', body=body, **kwargs),
            callback = callback
        )