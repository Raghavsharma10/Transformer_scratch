def create_tipo_roteiro(self):
        """Get an instance of tipo_roteiro services facade."""
        return TipoRoteiro(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)