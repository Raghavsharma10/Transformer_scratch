def add_capability(self, name, uri):
        """
        Add a capability of the RETS board
        :param name: The name of the capability
        :param uri: The capability URI given by the RETS board
        :return: None
        """

        parse_results = urlparse(uri)
        if parse_results.hostname is None:
            # relative URL given, so build this into an absolute URL
            login_url = self.capabilities.get('Login')
            if not login_url:
                logger.error("There is no login URL stored, so additional capabilities cannot be added.")
                raise ValueError("Cannot automatically determine absolute path for {0!s} given.".format(uri))

            parts = urlparse(login_url)
            port = ':{}'.format(parts.port) if parts.port else ''
            uri = parts.scheme + '://' + parts.hostname + port + '/' + uri.lstrip('/')

        self.capabilities[name] = uri