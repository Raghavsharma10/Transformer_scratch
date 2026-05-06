def create_doc(self,
            index,
            doc_type,
            body,
            doc_id   = None,
            params   = {},
            callback = None,
            **kwargs
        ):
        """
        Adds a typed JSON document in a specific index, making it searchable.
        Behind the scenes this method calls index(..., op_type='create')
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/docs-index_.html>`_
        :arg index: The name of the index
        :arg doc_type: The type of the document
        :arg doc_id: Document ID
        :arg body: The document
        :arg consistency: Explicit write consistency setting for the operation
        :arg id: Specific document ID (when the POST method is used)
        :arg parent: ID of the parent document
        :arg percolate: Percolator queries to execute while indexing the document
        :arg refresh: Refresh the index after performing the operation
        :arg replication: Specific replication type (default: sync)
        :arg routing: Specific routing value
        :arg timeout: Explicit operation timeout
        :arg timestamp: Explicit timestamp for the document
        :arg ttl: Expiration time for the document
        :arg version: Explicit version number for concurrency control
        :arg version_type: Specific version type
        """

        method = 'PUT' if doc_id else 'POST'

        query_params = ('consistency', 'op_type', 'parent', 'refresh',
            'replication', 'routing', 'timeout', 'timestamp', 'ttl', 'version',
            'version_type',
        )

        params = self._filter_params(query_params, params)

        url = self.mk_url(*[index, doc_type, doc_id], **params)

        self.client.fetch(
            self.mk_req(url, method=method, body=body, **kwargs),
            callback = callback
        )