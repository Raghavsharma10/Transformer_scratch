def verify_repository(self,
            repository,
            master_timeout = 10,
            timeout        = 10,
            body           = '',
            params         = {},
            callback       = None,
            **kwargs
        ):
        """
        Returns a list of nodes where repository was successfully verified or
        an error message if verification process failed.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/modules-snapshots.html>`_

        :arg repository: A repository name
        :arg master_timeout: Explicit operation timeout for connection to master
            node
        :arg timeout: Explicit operation timeout
        """

        query_params = ('master_timeout', 'timeout',)

        params = self._filter_params(query_params, params)

        url = self.mk_url(*['_snapshot', repository, '_verify'], **params)

        self.client.fetch(
            self.mk_req(url, body=body, method='POST', **kwargs),
            callback = callback
        )