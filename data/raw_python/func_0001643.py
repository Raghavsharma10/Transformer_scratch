def _connect(self):
        """Connects via SSH.
        """
        ssh = self._ssh_client()
        logger.debug("Connecting with %s",
                     ', '.join('%s=%r' % (k, v if k != "password" else "***")
                               for k, v in iteritems(self.destination)))
        ssh.connect(**self.destination)
        logger.debug("Connected to %s", self.destination['hostname'])
        self._ssh = ssh