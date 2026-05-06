def create_marca(self):
        """Get an instance of marca services facade."""
        return Marca(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)