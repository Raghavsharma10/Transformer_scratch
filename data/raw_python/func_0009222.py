def request(self, method, path, query=None, data=None, redirects=True):
        """
        Sends HTTP request to LendingClub.

        Parameters
        ----------
        method : {GET, POST, HEAD, DELETE}
            The HTTP method to use: GET, POST, HEAD or DELETE
        path : string
            The path that will be appended to the domain defined in :attr:`base_url`.
        query : dict
            A dictionary of query string parameters
        data : dict
            A dictionary of POST data values
        redirects : boolean
            True to follow redirects, False to return the original response from the server.

        Returns
        -------
        requests.Response
            A `requests.Response <http://docs.python-requests.org/en/latest/api/#requests.Response>`_ object
        """

        # Check session time
        self.__continue_session()

        try:
            url = self.build_url(path)
            method = method.upper()

            self.__log('{0} request to: {1}'.format(method, url))

            if method == 'POST':
                request = self.__session.post(url, params=query, data=data, allow_redirects=redirects)
            elif method == 'GET':
                request = self.__session.get(url, params=query, data=data, allow_redirects=redirects)
            elif method == 'HEAD':
                request = self.__session.head(url, params=query, data=data, allow_redirects=redirects)
            elif method == 'DELETE':
                request = self.__session.delete(url, params=query, data=data, allow_redirects=redirects)
            else:
                raise SessionError('{0} is not a supported HTTP method'.format(method))

            self.last_response = request

            self.__log('Status code: {0}'.format(request.status_code))

            # Update session time
            self.last_request_time = time.time()

        except (RequestException, ConnectionError, TooManyRedirects, HTTPError) as e:
            raise NetworkError('{0} failed to: {1}'.format(method, url), e)
        except Timeout:
            raise NetworkError('{0} request timed out: {1}'.format(method, url), e)

        return request