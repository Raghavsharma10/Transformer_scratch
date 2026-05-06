def list_passwords_search(self, searchstring):
        """List passwords with searchstring."""
        log.debug('List all passwords with: %s' % searchstring)
        return self.collection('passwords/search/%s.json' %
                               quote_plus(searchstring))