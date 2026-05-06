def create_divisao_dc(self):
        """Get an instance of divisao_dc services facade."""
        return DivisaoDc(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)