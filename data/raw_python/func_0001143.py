def open(self):
        """Ensure we have a connection to the email server.

        Returns whether or not a new connection was required (True or False).
        """
        if self.connection:
            # Nothing to do if the connection is already open.
            return False
        try:
            # If local_hostname is not specified, socket.getfqdn() gets used.
            # For performance, we use the cached FQDN for local_hostname.
            self.connection = smtplib.SMTP(
                self.host, self.port, local_hostname=DNS_NAME.get_fqdn()
            )
            if self.use_tls:
                self.connection.ehlo()
                self.connection.starttls()
                self.connection.ehlo()
            if self.username and self.password:
                self.connection.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(
                "Error trying to connect to server %s:%s: %s",
                self.host,
                self.port,
                e,
            )
            if not self.fail_silently:
                raise