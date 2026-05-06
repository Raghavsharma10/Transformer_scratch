def put_template(self, temp_id, body, params={}, callback=None, **kwargs):
        """
        Create a search template.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-template.html>`_
        :arg temp_id: Template ID
        :arg body: The document
        """

        url = self.mk_url(*['_search', 'template', temp_id])

        self.client.fetch(
            self.mk_req(url, method='PUT', body=body, **kwargs),
            callback = callback
        )