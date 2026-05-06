def create_usuario(self):
        """Get an instance of usuario services facade."""
        return Usuario(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)