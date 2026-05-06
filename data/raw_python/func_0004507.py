def create_grupo_virtual(self):
        """Get an instance of grupo_virtual services facade."""
        return GrupoVirtual(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)