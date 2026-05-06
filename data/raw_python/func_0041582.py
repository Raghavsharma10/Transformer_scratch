def _set_hostname_domain(self):
        """Extract hostname and domain"""
        self._hostname, _, self._domain = str(self._fqdn).partition('.')
        log.debug('Hostname: %s, Domain: %s' % (self._hostname, self._domain))