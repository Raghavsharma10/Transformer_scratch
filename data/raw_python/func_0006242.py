def start(self):
        """
            Launches a new POP3 client session on the server taken from the `self.options` dict.

        :param my_ip: IP of this Client itself
        """

        username = self.options['username']
        password = self.options['password']
        server_host = self.options['server']
        server_port = self.options['port']
        honeypot_id = self.options['honeypot_id']

        session = self.create_session(server_host, server_port, honeypot_id)

        try:
            logger.debug(
                'Sending {0} bait session to {1}:{2}. (bait id: {3})'.format('pop3', server_host, server_port,
                                                                             session.id))
            conn = poplib.POP3_SSL(server_host, server_port)
            session.source_port = conn.sock.getsockname()[1]

            banner = conn.getwelcome()
            session.protocol_data['banner'] = banner
            session.did_connect = True

            conn.user(username)
            conn.pass_(password)
            # TODO: Handle failed login
            session.add_auth_attempt('plaintext', True, username=username, password=password)
            session.did_login = True
            session.timestamp = datetime.utcnow()
        except Exception as err:
            logger.debug('Caught exception: {0} ({1})'.format(err, str(type(err))))
        else:
            list_entries = conn.list()[1]
            for entry in list_entries:
                index, _ = entry.split(' ')
                conn.retr(index)
                conn.dele(index)
            logger.debug('Found and deleted {0} messages on {1}'.format(len(list_entries), server_host))
            conn.quit()
            session.did_complete = True
        finally:
            session.alldone = True
            session.end_session()