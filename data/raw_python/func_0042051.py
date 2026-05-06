def path_requests(self, path):
        """
        Returns a Resource instace that will have attributes, one for each of the http-methods
        supported on that path.  For example:

            >>> hcl_api = client.path_requests('/api/hcl/{id}')
            >>> dir(hcl_api)
            [u'delete', u'get', u'put']

            >>> resp, ok = hcl_api.get(id='Arista_vEOS')

        Parameters
        ----------
        path : str
            The API path

        Returns
        -------
        Resource
            instance that has attributes for methods available.
        """
        path_spec = self.client.origin_spec['paths'].get(path)
        if not path_spec:
            raise RuntimeError("no path found for: %s" % path)

        get_for_meth = self.client.swagger_spec.get_op_for_request
        rsrc = BravadoResource(name=path, ops={
            method: get_for_meth(method, path)
            for method in path_spec.keys()})

        return RequestFactory.Resource(self.client, ResourceDecorator(rsrc))