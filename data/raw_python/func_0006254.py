def start(self):
        """
            Launches a new SMTP client session on the server taken from the `self.options` dict.

        :param my_ip: IP of this Client itself
        """

        username = self.options['username']
        password = self.options['password']
        server_host = self.options['server']
        server_port = self.options['port']
        honeypot_id = self.options['honeypot_id']

        session = self.create_session(server_host, server_port, honeypot_id)

        logger.debug(
            'Sending {0} bait session to {1}:{2}. (bait id: {3})'.format('smtp', server_host, server_port, session.id))

        try:
            self.connect()
            session.did_connect = True
            session.source_port = self.client.sock.getsockname()[1]
            self.login(username, password)

            # TODO: Handle failed login
            # TODO: password='' is sillly fix, this needs to be fixed server side...
            session.add_auth_attempt('plaintext', True, username=username, password='')
            session.did_login = True

        except smtplib.SMTPException as error:
            logger.debug('Caught exception: {0} ({1})'.format(error, str(type(error))))
        else:
            while self.sent_mails <= self.max_mails:
                from_addr, to_addr, mail_body = self.get_one_mail()
                try:
                    if from_addr and to_addr and isinstance(mail_body, str):
                        self.client.sendmail(from_addr, to_addr, mail_body)
                    else:
                        continue
                except TypeError as e:
                    logger.debug('Malformed email in mbox archive, skipping.')
                    continue
                else:
                    self.sent_mails += 1
                    logger.debug('Sent mail from ({0}) to ({1})'.format(from_addr, to_addr))
                time.sleep(1)
            self.client.quit()
            session.did_complete = True
        finally:
            logger.debug('SMTP Session complete.')
            session.alldone = True
            session.end_session()
            self.client.close()