def list_mypasswords_search(self, searchstring):
        """List my passwords with searchstring."""
        # http://teampasswordmanager.com/docs/api-my-passwords/#list_passwords
        log.debug('List MyPasswords with %s' % searchstring)
        return self.collection('my_passwords/search/%s.json' %
                               quote_plus(searchstring))