def start(self):
        """
            Launches a new Telnet client session on the server taken from the `self.options` dict.

        :param my_ip: IP of this Client itself
        """

        login = self.options['username']
        password = self.options['password']
        server_host = self.options['server']
        server_port = self.options['port']
        honeypot_id = self.options['honeypot_id']
        command_limit = random.randint(6, 11)

        session = self.create_session(server_host, server_port, honeypot_id)
        self.sessions[session.id] = session
        logger.debug(
            'Sending telnet bait session to {0}:{1}. (bait id: {2})'.format(server_host, server_port, session.id))

        try:
            self.connect()
            self.login(login, password)

            session.add_auth_attempt('plaintext', True, username=login, password=password)

            session.did_connect = True
            session.source_port = self.client.sock.getsockname()[1]
            session.did_login = True
        except InvalidLogin:
            logger.debug('Telnet session could not login. ({0})'.format(session.id))
            session.did_login = False
        except Exception as err:
            logger.debug('Caught exception: {0} {1}'.format(err, str(err), exc_info=True))
        else:
            command_count = 0
            while command_count < command_limit:
                command_count += 1
                self.sense()
                comm, param = self.decide()
                self.act(comm, param)
                gevent.sleep(random.uniform(0.4, 5.6))
            self.act('logout')
            session.did_complete = True
        finally:
            session.alldone = True
            session.end_session()
            if self.client:
                self.client.close()