def create_equipamento(self):
        """Get an instance of equipamento services facade."""
        return Equipamento(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)