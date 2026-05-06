def execute(self, **kwargs):
        """
        Calls the FritzBox action and returns a dictionary with the arguments.
        """
        headers = self.header.copy()
        headers['soapaction'] = '%s#%s' % (self.service_type, self.name)
        data = self.envelope.strip() % self._body_builder(kwargs)
        url = 'http://%s:%s%s' % (self.address, self.port, self.control_url)
        auth = None
        if self.password:
            auth=HTTPDigestAuth(self.user, self.password)
        response = requests.post(url, data=data, headers=headers, auth=auth)
        # lxml needs bytes, therefore response.content (not response.text)
        result = self.parse_response(response.content)
        return result