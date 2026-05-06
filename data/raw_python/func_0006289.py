def delete(self, endpoint, **kwargs):
        """ Send HTTP DELETE to the endpoint.

        :arg str endpoint: The endpoint to send to.

        :returns:
            JSON decoded result.

        :raises:
            requests.RequestException on timeout or connection error.

        """
        kwargs.update(self.kwargs.copy())
        response = requests.delete(self.make_url(endpoint), **kwargs)
        return _decode_response(response)