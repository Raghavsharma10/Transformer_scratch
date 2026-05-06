def _single_request(self, method, *args, **kwargs):
        """Make a single request to the fleet API endpoint

        Args:
            method (str): A dot delimited string indicating the method to call.  Example: 'Machines.List'
            *args: Passed directly to the method being called.
            **kwargs: Passed directly to the method being called.

        Returns:
            dict: The response from the method called.

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400
        """

        # The auto generated client binding require instantiating each object you want to call a method on
        # For example to make a request to /machines for the list of machines you would do:
        # self._service.Machines().List(**kwargs)
        # This code iterates through the tokens in `method` and instantiates each object
        # Passing the `*args` and `**kwargs` to the final method listed

        # Start here
        _method = self._service

        # iterate over each token in the requested method
        for item in method.split('.'):

            # if it's the end of the line, pass our argument
            if method.endswith(item):
                _method = getattr(_method, item)(*args, **kwargs)
            else:
                # otherwise, just create an instance and move on
                _method = getattr(_method, item)()

        # Discovered endpoints look like r'$ENDPOINT/path/to/method' which isn't a valid URI
        # Per the fleet API documentation:
            # "Note that this discovery document intentionally ships with an unusable rootUrl;
            # clients must initialize this as appropriate."

        # So we follow the documentation, and replace the token with our actual endpoint
        _method.uri = _method.uri.replace('$ENDPOINT', self._endpoint)

        # Execute the method and return it's output directly
        try:
            return _method.execute(http=self._http)
        except googleapiclient.errors.HttpError as exc:
            response = json.loads(exc.content.decode('utf-8'))['error']

            raise APIError(code=response['code'], message=response['message'], http_error=exc)