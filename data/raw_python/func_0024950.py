def get_collection(self, collection, filter=None, fields=None,
            page_size=None):
        """
        Returns a specific collection from the asset service with
        the given collection endpoint.

        Supports passing through parameters such as...
        - filters such as "name=Vesuvius" following GEL spec
        - fields such as "uri,description" comma delimited
        - page_size such as "100" (the default)

        """
        params = {}
        if filter:
            params['filter'] = filter
        if fields:
            params['fields'] = fields
        if page_size:
            params['pageSize'] = page_size

        uri = self.uri + '/v1' + collection
        return self.service._get(uri, params=params)