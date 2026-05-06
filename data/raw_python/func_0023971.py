def _authorize_new_tokens(self):
        '''
        Stands up a new localhost http server and retrieves new OAuth2 access
        tokens from the Coursera OAuth2 server.
        '''
        logging.info('About to request new OAuth2 tokens from Coursera.')
        # Attempt to request new tokens from Coursera via the browser.
        state_token = uuid.uuid4().hex
        authorization_url = self._build_authorizaton_url(state_token)

        sys.stdout.write(
            'Please visit the following URL to authorize this app:\n')
        sys.stdout.write('\t%s\n\n' % authorization_url)
        if _platform == 'darwin':
            # OS X -- leverage the 'open' command present on all modern macs
            sys.stdout.write(
                'Mac OS X detected; attempting to auto-open the url '
                'in your default browser...\n')
            try:
                subprocess.check_call(['open', authorization_url])
            except:
                logging.exception('Could not call `open %(url)s`.',
                                  url=authorization_url)

        if self.local_webserver_port is not None:
            # Boot up a local webserver to retrieve the response.
            server_address = ('', self.local_webserver_port)
            code_holder = CodeHolder()

            local_server = BaseHTTPServer.HTTPServer(
                server_address,
                _make_handler(state_token, code_holder))

            while not code_holder.has_code():
                local_server.handle_request()
            coursera_code = code_holder.code
        else:
            coursera_code = raw_input('Please enter the code from Coursera: ')

        form_data = {
            'code': coursera_code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self._redirect_uri,
            'grant_type': 'authorization_code',
        }
        return self._request_tokens_from_token_endpoint(form_data)