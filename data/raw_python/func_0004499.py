def create_equipamento_acesso(self):
        """Get an instance of equipamento_acesso services facade."""
        return EquipamentoAcesso(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)