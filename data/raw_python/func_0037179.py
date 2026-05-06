def update(self, source_id, source_type, name, resources, auth, parameters=None, validate=True):
        """ Update a managed source

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourceupdate

            :param source_type: data source name e.g. facebook_page, googleplus, instagram, yammer
            :type source_type: str
            :param name: name to use to identify the managed source being created
            :type name: str
            :param resources: list of source-specific config dicts
            :type resources: list
            :param auth: list of source-specific authentication dicts
            :type auth: list
            :param parameters: (optional) dict with config information on how to treat each resource
            :type parameters: dict
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        assert resources, "Need at least one resource"
        assert auth, "Need at least one authentication token"
        params = {'id': source_id, 'source_type': source_type, 'name': name, 'resources': resources, 'auth': auth, 'validate': validate}
        if parameters:
            params['parameters'] = parameters

        return self.request.post('update', params)