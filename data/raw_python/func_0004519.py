def create_permission(self):
        """Get an instance of permission services facade."""
        return Permission(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)