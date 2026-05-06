def put(self, endpoint, **kwargs):
        """ Send HTTP PUT to the endpoint.

        :arg str endpoint: The endpoint to send to.

        :returns:
            JSON decoded result.

        :raises:
            requests.RequestException on timeout or connection error.

        """
        kwargs.update(self.kwargs.copy())
        if "data" in kwargs:
            kwargs["headers"].update(
                {"Content-Type": "application/json;charset=UTF-8"})
        response = requests.put(self.make_url(endpoint), **kwargs)
        return _decode_response(response)