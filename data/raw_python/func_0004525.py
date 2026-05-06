def create_tipo_acesso(self):
        """Get an instance of tipo_acesso services facade."""
        return TipoAcesso(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)