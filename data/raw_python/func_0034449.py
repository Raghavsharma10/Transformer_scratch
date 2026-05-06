def scroll(self, scroll_id, params={}, callback=None, **kwargs):
        """
        Scroll a search request created by specifying the scroll parameter.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-request-scroll.html>`_
        :arg scroll_id: The scroll ID
        :arg scroll: Specify how long a consistent view of the index should be
            maintained for scrolled search
        """

        query_params = ('scroll',)

        params = self._filter_params(query_params, params)

        url = self.mk_url(*['/_search/scroll'], **params)

        self.client.fetch(
            self.mk_req(url, method='GET', body=scroll_id, **kwargs),
            callback = callback
        )