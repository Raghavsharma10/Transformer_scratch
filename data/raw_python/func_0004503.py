def create_filter(self):
        """Get an instance of filter services facade."""
        return Filter(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)