def create_tipo_equipamento(self):
        """Get an instance of tipo_equipamento services facade."""
        return TipoEquipamento(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)