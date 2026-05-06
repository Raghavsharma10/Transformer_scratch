def _do_connect(self):
        """ Connect to the remote. """
        self.load_system_host_keys()
        if self.username is None or self.port is None:
            self._configure()
        try:
            self.connect(hostname=self.hostname,
                         port=self.port,
                         username=self.username,
                         key_filename=self.key_filename,
                         sock=self.proxy)
        except socket.error as e:
            raise GerritError("Failed to connect to server: %s" % e)

        try:
            version_string = self._transport.remote_version
            pattern = re.compile(r'^.*GerritCodeReview_([a-z0-9-\.]*) .*$')
            self.remote_version = _extract_version(version_string, pattern)
        except AttributeError:
            self.remote_version = None