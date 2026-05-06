def put_script(self, lang, script_id, body, params={}, callback=None, **kwargs):
        """
        Create a script in given language with specified ID.
        `<http://www.elasticsearch.org/guide/en/elasticsearch/reference/current/modules-scripting.html>`_
        :arg lang: Script language
        :arg script_id: Script ID
        :arg body: The document
        :arg op_type: Explicit operation type, default u'index'
        :arg version: Explicit version number for concurrency control
        :arg version_type: Specific version type
        """
        query_params = ('op_type', 'version', 'version_type',)

        params = self._filter_params(query_params, params)

        url = self.mk_url(*['_scripts', lang, script_id], **params)

        self.client.fetch(
            self.mk_req(url, method='PUT', body=body, **kwargs),
            callback = callback
        )