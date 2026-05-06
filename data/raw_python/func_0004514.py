def create_dhcprelay_ipv6(self):
        """Get an instance of DHCPRelayIPv6 services facade."""
        return DHCPRelayIPv6(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)