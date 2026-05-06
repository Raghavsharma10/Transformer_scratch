def post_request(self, container, resource=None, params=None, accept=None):
        """Send a POST request."""
        url = self.make_url(container, resource)
        headers = self._make_headers(accept)

        try:
            rsp = requests.post(url, data=params, headers=headers,
                                verify=self._verify, timeout=self._timeout)
        except requests.exceptions.ConnectionError as e:
            RestHttp._raise_conn_error(e)

        if self._dbg_print:
            self.__print_req('POST', rsp.url, headers, params)

        return self._handle_response(rsp)