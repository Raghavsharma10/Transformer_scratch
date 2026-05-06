def create(self, source_type, name, resources, auth=None, parameters=None, validate=True):
        """ Create a managed source

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourcecreate

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
            :param validate: bool to determine if validation should be performed on the source
            :type validate: bool
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        assert resources, "Need at least one resource"

        params = {
            'source_type': source_type,
            'name': name,
            'resources': resources,
            'validate': validate
        }

        if auth:
            params['auth'] = auth
        if parameters:
            params['parameters'] = parameters

        return self.request.post('create', params)