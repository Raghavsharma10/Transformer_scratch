def create_ambiente(self):
        """Get an instance of ambiente services facade."""
        return Ambiente(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)