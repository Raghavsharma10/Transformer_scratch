def connect(self):
        """
            Connect to the SMTP server.
        """
        # TODO: local_hostname should be configurable
        self.client = smtplib.SMTP(self.options['server'], self.options['port'],
                                   local_hostname='local.domain', timeout=15)