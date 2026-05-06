def delete_template(self, temp_id=None, params={}, callback=None, **kwargs):
        """
        Delete a search template.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/search-template.html>`_
        :arg temp_id: Template ID
        """

        url = self.mk_url(*['_search', 'template', temp_id])

        self.client.fetch(
            self.mk_req(url, method='DELETE', **kwargs),
            callback = callback
        )