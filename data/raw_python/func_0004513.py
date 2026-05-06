def create_dhcprelay_ipv4(self):
        """Get an instance of DHCPRelayIPv4 services facade."""
        return DHCPRelayIPv4(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)