def create_direito_grupo_equipamento(self):
        """Get an instance of direito_grupo_equipamento services facade."""
        return DireitoGrupoEquipamento(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)