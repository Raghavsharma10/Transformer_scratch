def create_system(self):
        """Get an instance of Api System Variables services facade."""
        return System(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)