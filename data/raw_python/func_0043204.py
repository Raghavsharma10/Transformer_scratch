def _put_or_post_multipart(self, method, url, data):
        """
        encodes the data as a multipart form and PUTs or POSTs to the url
        the response is parsed as JSON and the returns the resulting data structure
        """
        fields = []
        files = []
        for key, value in data.items():
            if type(value) == file:
                files.append((key, value.name, value.read()))
            else:
                fields.append((key, value))
        content_type, body = _encode_multipart_formdata(fields, files)
        if self.parsed_endpoint.scheme == 'https':
            h = httplib.HTTPS(self.parsed_endpoint.netloc)
        else:
            h = httplib.HTTP(self.parsed_endpoint.netloc)
        h.putrequest(method, url)
        h.putheader('Content-Type', content_type)
        h.putheader('Content-Length', str(len(body)))
        h.putheader('Accept', 'application/json')
        h.putheader('User-Agent', USER_AGENT)
        h.putheader(API_TOKEN_HEADER_NAME, self.api_token)
        if self.api_version in ['0.1', '0.01a']:
            h.putheader(API_VERSION_HEADER_NAME, self.api_version)
        h.endheaders()
        h.send(body)
        errcode, errmsg, headers = h.getreply()
        if errcode not in [200, 202]:
            raise IOError('Response to %s to URL %s was status code %s: %s' % (method, url, errcode, h.file.read()))
        return json.loads(h.file.read())