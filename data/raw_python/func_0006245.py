def start(self):
        """
            Launches a new SSH client session on the server taken from the `self.options` dict.

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
            'Sending ssh bait session to {0}:{1}. (bait id: {2})'.format(server_host, server_port, session.id))
        try:
            self.connect_login()
            session.did_connect = True
            # TODO: Handle failed login
            session.add_auth_attempt('plaintext', True, username=username, password=password)
            session.did_login = True
        except (SSHException, AuthenticationFailed) as err:
            logger.debug('Caught exception: {0} ({1})'.format(err, str(type(err))))
        else:
            command_count = 0
            command_limit = random.randint(6, 11)
            while command_count < command_limit:
                command_count += 1
                self.sense()
                comm, param = self.decide()
                self.act(comm, param)
                gevent.sleep(random.uniform(0.4, 5.6))
            self.logout()
            session.did_complete = True
        finally:
            session.alldone = True
            session.end_session()
            self.comm_chan.close()