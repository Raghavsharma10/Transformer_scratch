def _api_call(self, method, target, args=None):
        """Create a Request object and execute the call to the API Server.
        Parameters
            method (str)
                HTTP request (e.g. 'POST').
            target (str)
                The target URL with leading slash (e.g. '/v1/products').
            args (dict)
                Optional dictionary of arguments to attach to the request.
        Returns
            (Response)
                The server's response to an HTTP request.
        """
        self.refresh_oauth_credential()

        request = Request(
            auth_session=self.session,
            api_host=self.api_host,
            method=method,
            path=target,
            args=args,
        )

        return request.execute()