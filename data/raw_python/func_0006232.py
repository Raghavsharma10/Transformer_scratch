def start(self):

        """
            Launches a new FTP client session on the server taken from the `self.options` dict.

        :param my_ip: IP of this Client itself
        """
        username = self.options['username']
        password = self.options['password']
        server_host = self.options['server']
        server_port = self.options['port']
        honeypot_id = self.options['honeypot_id']
        command_limit = random.randint(6, 11)

        session = self.create_session(server_host, server_port, honeypot_id)

        self.sessions[session.id] = session
        logger.debug(
            'Sending {0} bait session to {1}:{2}. (bait id: {3})'.format('ftp', server_host, server_port, session.id))

        self.file_list = []
        try:
            self.connect()
            session.did_connect = True

            # TODO: Catch login failure
            self.login(username, password)
            session.add_auth_attempt('plaintext', True, username=username, password=password)

            session.did_login = True
            session.timestamp = datetime.utcnow()
        except ftplib.error_perm as err:
            logger.debug('Caught exception: {0} ({1})'.format(err, str(type(err))))
        except socket.error as err:
            logger.debug('Error while communicating: {0} ({1})'.format(err, str(type(err))))
        else:
            command_count = 0
            while command_count <= command_limit:
                command_count += 1
                try:
                    self.sense()
                    cmd, param = self.decide()
                    self.act(cmd, param)
                    gevent.sleep(random.uniform(0, 3))
                except IndexError:  # This means we hit an empty folder, or a folder with only files.
                    continue
            session.did_complete = True
        finally:
            if self.client.sock is not None:
                # will close socket
                self.client.quit()
            session.alldone = True
        session.end_session()