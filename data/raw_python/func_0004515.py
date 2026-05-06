def create_network(self):
        """Get an instance of vlan services facade."""
        return Network(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)