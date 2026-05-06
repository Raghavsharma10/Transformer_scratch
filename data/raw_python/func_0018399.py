def __get_response(self, uri, params=None, method="get", stream=False):
        """Creates a response object with the given params and option

            Parameters
            ----------
            url : string
                The full URL to request.
            params: dict
                A list of parameters to send with the request.  This
                will be sent as data for methods that accept a request
                body and will otherwise be sent as query parameters.
            method : str
                The HTTP method to use.
            stream : bool
                Whether to stream the response.

            Returns a requests.Response object.
        """
        if not hasattr(self, "session") or not self.session:
            self.session = requests.Session()
            if self.access_token:
                self.session.headers.update(
                    {'Authorization': 'Bearer {}'.format(self.access_token)}
                )

        # Remove empty params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        kwargs = {
            "url": uri,
            "verify": True,
            "stream": stream
        }

        kwargs["params" if method == "get" else "data"] = params

        return getattr(self.session, method)(**kwargs)