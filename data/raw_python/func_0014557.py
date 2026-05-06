def head_request(self, container, resource=None):
        """Send a HEAD request."""
        url = self.make_url(container, resource)
        headers = self._make_headers(None)

        try:
            rsp = requests.head(url, headers=self._base_headers,
                                verify=self._verify, timeout=self._timeout)
        except requests.exceptions.ConnectionError as e:
            RestHttp._raise_conn_error(e)

        if self._dbg_print:
            self.__print_req('HEAD', rsp.url, headers, None)

        return rsp.status_code