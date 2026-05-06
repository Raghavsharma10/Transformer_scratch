def create_permissao_administrativa(self):
        """Get an instance of permissao_administrativa services facade."""
        return PermissaoAdministrativa(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)