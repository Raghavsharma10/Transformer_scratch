def _get_data(self, url, accept=None):
        """
        GETs the resource at url and returns the raw response
        If the accept parameter is not None, the request passes is as the Accept header
        """
        if self.parsed_endpoint.scheme == 'https':
            conn = httplib.HTTPSConnection(self.parsed_endpoint.netloc)
        else:
            conn = httplib.HTTPConnection(self.parsed_endpoint.netloc)
        head = {
            "User-Agent": USER_AGENT,
            API_TOKEN_HEADER_NAME: self.api_token,
        }
        if self.api_version in ['0.1', '0.01a']:
            head[API_VERSION_HEADER_NAME] = self.api_version
        if accept:
            head['Accept'] = accept
        conn.request("GET", url, "", head)
        resp = conn.getresponse()
        self._handle_response_errors('GET', url, resp)
        content_type = resp.getheader('content-type')
        if 'application/json' in content_type:
            return json.loads(resp.read())
        return resp.read()