def create_equipamento_ambiente(self):
        """Get an instance of equipamento_ambiente services facade."""
        return EquipamentoAmbiente(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)