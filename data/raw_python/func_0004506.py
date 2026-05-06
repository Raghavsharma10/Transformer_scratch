def create_grupo_usuario(self):
        """Get an instance of grupo_usuario services facade."""
        return GrupoUsuario(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)