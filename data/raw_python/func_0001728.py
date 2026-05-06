def connect(self):
        # type: () -> None
        """
        Connect to server

        Returns:
            None

        """
        if self.connection_type.lower() == 'ssl':
            self.server = smtplib.SMTP_SSL(host=self.host, port=self.port, local_hostname=self.local_hostname,
                                           timeout=self.timeout, source_address=self.source_address)
        elif self.connection_type.lower() == 'lmtp':
            self.server = smtplib.LMTP(host=self.host, port=self.port, local_hostname=self.local_hostname,
                                       source_address=self.source_address)
        else:
            self.server = smtplib.SMTP(host=self.host, port=self.port, local_hostname=self.local_hostname,
                                       timeout=self.timeout, source_address=self.source_address)
        self.server.login(self.username, self.password)