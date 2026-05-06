def _request(self, http_verb, url, data=None, authenticated=True):
        """
        Perform an HTTP request.

        See https://docs.python.org/3/library/json.html#json-to-py-table for the http response object.

        :param http_verb: the HTTP verb (GET, POST, PUT, …)
        :type http_verb: string
        :param url: the path to the resource queried
        :type url: string
        :param data: the request content
        :type data: dict
        :param authenticated: the request should be authenticated
        :type authenticated: bool

        :return: http response, http status
        :rtype tuple(object, int)
        """
        user_agent = ('PayPlug-Python/{lib_version} (Python/{python_version}; '
                      '{request_library})'
                      .format(lib_version=__version__,
                              python_version=_get_python_version_string(),
                              request_library=self._request_handler.get_useragent_string()))
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': user_agent,
        }
        if authenticated:
            headers['Authorization'] = 'Bearer ' + self._secret_key

        requestor = self._request_handler()
        response, status, _ = requestor.do_request(http_verb, url, headers, data)

        # Since Python 3.2+, response body is a bytes-like object. We have to decode it to a string.
        if isinstance(response, six.binary_type):
            response = response.decode('utf-8')

        if not 200 <= status < 300:
            raise exceptions.HttpError.map_http_status_to_exception(status)(http_response=response,
                                                                            http_status=status)

        try:
            response_object = json.loads(response)
        except ValueError:
            raise exceptions.UnexpectedAPIResponseException(http_response=response, http_status=status)

        return response_object, status