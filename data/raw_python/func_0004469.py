def create_ambiente_logico(self):
        """Get an instance of ambiente_logico services facade."""
        return AmbienteLogico(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)