def create(self, term, options):
        """Create a monitor using passed configuration."""
        if not self._state:
            raise InvalidState("State was not properly obtained from the app")
        options['action'] = 'CREATE'
        payload = self._build_payload(term, options)
        url = self.ALERTS_CREATE_URL.format(requestX=self._state[3])
        self._log.debug("Creating alert using: %s" % url)
        params = json.dumps(payload, separators=(',', ':'))
        data = {'params': params}
        response = self._session.post(url, data=data, headers=self.HEADERS)
        if response.status_code != 200:
            raise ActionError("Failed to create monitor: %s"
                              % response.content)
        if options.get('exact', False):
            term = "\"%s\"" % term
        return self.list(term)