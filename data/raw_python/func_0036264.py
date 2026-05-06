def request(self, path, method='GET', headers=None, **kwargs):
        """Perform a HTTP request.

        Given a relative Bugzilla URL path, an optional request method,
        and arguments suitable for requests.Request(), perform a
        HTTP request.
        """
        headers = {} if headers is None else headers.copy()
        headers["User-Agent"] = "Bugsy"
        kwargs['headers'] = headers
        url = '%s/%s' % (self.bugzilla_url, path)
        return self._handle_errors(self.session.request(method, url, **kwargs))