def create_usuario_grupo(self):
        """Get an instance of usuario_grupo services facade."""
        return UsuarioGrupo(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)