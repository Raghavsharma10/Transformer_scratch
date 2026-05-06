def start(self):

        """
            Launches a new HTTP client session on the server taken from the `self.options` dict.

        :param my_ip: IP of this Client itself
        """
        username = self.options['username']
        password = self.options['password']
        server_host = self.options['server']
        server_port = self.options['port']
        honeypot_id = self.options['honeypot_id']

        session = self.create_session(server_host, server_port, honeypot_id)

        self.sessions[session.id] = session

        logger.debug(
            'Sending {0} bait session to {1}:{2}. (bait id: {3})'.format('http', server_host, server_port, session.id))

        try:
            url = self._make_url(server_host, '/index.html', server_port)
            response = self.client.get(url, auth=HTTPBasicAuth(username, password), verify=False)
            session.did_connect = True
            if response.status_code == 200:
                session.add_auth_attempt('plaintext', True, username=username, password=password)
                session.did_login = True
            else:
                session.add_auth_attempt('plaintext', False, username=username, password=password)

            links = self._get_links(response)
            while self.sent_requests <= self.max_requests and links:
                url = random.choice(links)
                response = self.client.get(url, auth=HTTPBasicAuth(username, password), verify=False)
                links = self._get_links(response)

            session.did_complete = True
        except Exception as err:
            logger.debug('Caught exception: {0} ({1})'.format(err, str(type(err))))
        finally:
            session.alldone = True
            session.end_session()
            self.client.close()