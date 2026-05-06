def request(self, url, params=None, data=None, method='GET'):
        """
        Perform an HTTP request and return the response body as a decoded JSON
        value

        :param str url: the URL to make the request of.  If ``url`` begins with
            a forward slash, :attr:`endpoint` is prepended to it; otherwise,
            ``url`` is treated as an absolute URL.
        :param dict params: parameters to add to the URL's query string
        :param data: a value to send in the body of the request.  If ``data``
            is not a string, it will be serialized as JSON before sending;
            either way, the :mailheader:`Content-Type` header of the request
            will be set to :mimetype:`application/json`.  Note that a ``data``
            value of `None` means "Don't send any data"; to send an actual
            `None` value, convert it to JSON (i.e., the string ``"null"``)
            first.
        :param str method: the HTTP method to use: ``"GET"``, ``"POST"``,
            ``"PUT"``, or ``"DELETE"`` (case-insensitive); default: ``"GET"``
        :return: a decoded JSON value, or `None` if no data was returned
        :rtype: `list` or `dict` (depending on the request) or `None`
        :raises ValueError: if ``method`` is an invalid value
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if url.startswith('/'):
            url = self.endpoint + url
        attrs = {
            "headers": {"Authorization": "Bearer " + self.api_token},
            "params": params if params is not None else {},
            "timeout": self.timeout,
        }
        method = method.upper()
        if data is not None:
            if not isinstance(data, string_types):
                data = json.dumps(data, cls=DOEncoder)
            attrs["data"] = data
            attrs["headers"]["Content-Type"] = "application/json"
        if method == 'GET':
            r = self.session.get(url, **attrs)
        elif method == 'POST':
            r = self.session.post(url, **attrs)
        elif method == 'PUT':
            r = self.session.put(url, **attrs)
        elif method == 'DELETE':
            r = self.session.delete(url, **attrs)
        else:
            raise ValueError('Unrecognized HTTP method: ' + repr(method))
        self.last_response = r
        self.last_meta = None
        if not r.ok:
            raise DOAPIError(r)
        if r.text.strip():
            # Even when returning "no content", the API can still return
            # whitespace.
            response = r.json()
            try:
                self.last_meta = response["meta"]
            except (KeyError, TypeError):
                pass
            return response