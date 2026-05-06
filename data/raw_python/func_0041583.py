def _set_ip(self):
        """Resolve FQDN to IP address"""
        self._ip = socket.gethostbyname(self._fqdn)
        log.debug('IP: %s' % self._ip)