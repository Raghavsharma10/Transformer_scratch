def create_tipo_rede(self):
        """Get an instance of tipo_rede services facade."""
        return TipoRede(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)