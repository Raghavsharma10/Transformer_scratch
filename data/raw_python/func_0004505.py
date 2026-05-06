def create_grupo_l3(self):
        """Get an instance of grupo_l3 services facade."""
        return GrupoL3(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)