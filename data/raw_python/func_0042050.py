def command_request(self, method, path):
        """
        Returns a callable request for a given http method and API path.
        You can then use this request to execute the command, and get
        the response value:

            >>> rqst = client.command_request('get', '/api/hcl')
            >>> resp, ok = rqst()

        Parameters
        ----------
        method : str
            the http method value, ['get', 'put', 'post', ...]

        path : str
            the API route string value, for example:
            "/api/resources/vlan-pools/{id}"

        Returns
        -------
        Request
            The request instance you can then use to exeute the command.
        """
        op = self.client.swagger_spec.get_op_for_request(method, path)
        if not op:
            raise RuntimeError(
                'no command found for (%s, %s)' % (method, path))

        return Request(self.client, CallableOperation(op))