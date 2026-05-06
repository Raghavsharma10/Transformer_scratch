def convert_user_to_ldap(self, ID, DN):
        """Convert a normal user to a LDAP user."""
        # http://teampasswordmanager.com/docs/api-users/#convert_to_ldap
        data = {'login_dn': DN}
        log.info('Convert User %s to LDAP DN %s' % (ID, DN))
        self.put('users/%s/convert_to_ldap.json' % ID, data)