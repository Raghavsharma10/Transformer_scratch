def create_grupo_equipamento(self):
        """Get an instance of grupo_equipamento services facade."""
        return GrupoEquipamento(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)