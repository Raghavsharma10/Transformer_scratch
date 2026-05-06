def create_api_network_ipv6(self):
        """Get an instance of Api Networkv6 services facade."""

        return ApiNetworkIPv6(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)