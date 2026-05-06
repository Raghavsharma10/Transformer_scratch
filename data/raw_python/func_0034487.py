def create_snapshot(self, repository, snapshot, body, params={}, callback=None, **kwargs):
        """
        Create a snapshot in repository
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/master/modules-snapshots.html>`_

        :arg repository: A repository name
        :arg snapshot: A snapshot name
        :arg body: The snapshot definition
        :arg master_timeout: Explicit operation timeout for connection to master
            node
        :arg wait_for_completion: Should this request wait until the operation
            has completed before returning, default False
        """

        query_params = ('master_timeout', 'wait_for_completion',)

        params = self._filter_params(query_params, params)

        url = self.mk_url(*['_snapshot', repository, snapshot], **params)

        self.client.fetch(
            self.mk_req(url, body=body, method='PUT', **kwargs),
            callback = callback
        )