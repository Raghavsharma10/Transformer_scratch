def create_modelo(self):
        """Get an instance of modelo services facade."""
        return Modelo(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)