def update_custom_fields_of_password(self, ID, data):
        """Update custom fields definitions of a password."""
        # http://teampasswordmanager.com/docs/api-passwords/#update_cf_password
        log.info('Update custom fields of password %s with %s' % (ID, data))
        self.put('passwords/%s/custom_fields.json' % ID, data)