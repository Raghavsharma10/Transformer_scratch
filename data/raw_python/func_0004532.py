def create_vlan(self):
        """Get an instance of vlan services facade."""
        return Vlan(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)