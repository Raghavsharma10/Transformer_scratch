def identified(self, res):
        '''Handle a response to our 'identify' command. Returns response'''
        # If they support it, they should give us a JSON blob which we should
        # inspect.
        try:
            res.data = json.loads(res.data)
            self._identify_response = res.data
            logger.info('Got identify response: %s', res.data)
        except:
            logger.warn('Server does not support feature negotiation')
            self._identify_response = {}

        # Save our max ready count unless it's not provided
        self.max_rdy_count = self._identify_response.get(
            'max_rdy_count', self.max_rdy_count)
        if self._identify_options.get('tls_v1', False):
            if not self._identify_response.get('tls_v1', False):
                raise UnsupportedException(
                    'NSQd instance does not support TLS')
            else:
                self._socket = TLSSocket.wrap_socket(self._socket)

        # Now is the appropriate time to send auth
        if self._identify_response.get('auth_required', False):
            if not self._auth_secret:
                raise UnsupportedException(
                    'Auth required but not provided')
            else:
                self.auth(self._auth_secret)
                # If we're not talking over TLS, warn the user
                if not self._identify_response.get('tls_v1', False):
                    logger.warn('Using AUTH without TLS')
        elif self._auth_secret:
            logger.warn('Authentication secret provided but not required')
        return res