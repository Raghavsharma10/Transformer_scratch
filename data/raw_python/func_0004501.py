def create_equipamento_roteiro(self):
        """Get an instance of equipamento_roteiro services facade."""
        return EquipamentoRoteiro(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)