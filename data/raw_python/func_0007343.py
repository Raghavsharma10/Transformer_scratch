def unlock_password(self, ID, reason):
        """Unlock a password."""
        # http://teampasswordmanager.com/docs/api-passwords/#unlock_password
        log.info('Unlock password %s, Reason: %s' % (ID, reason))
        self.unlock_reason = reason
        self.put('passwords/%s/unlock.json' % ID)