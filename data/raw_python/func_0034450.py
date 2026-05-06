def clear_scroll(self,
            scroll_id = None,
            body      = '',
            params    = {},
            callback  = None,
            **kwargs
        ):
        """
        Clear the scroll request created by specifying the scroll parameter to
        search.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-request-scroll.html>`_
        :arg scroll_id: The scroll ID or a list of scroll IDs
        :arg body: A comma-separated list of scroll IDs to clear if none was
            specified via the scroll_id parameter
        """

        url = self.mk_url(*['_search', 'scroll', scroll_id])

        self.client.fetch(
            self.mk_req(url, method='DELETE', body=body, **kwargs),
            callback = callback
        )