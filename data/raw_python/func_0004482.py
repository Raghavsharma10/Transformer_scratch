def create_api_network_ipv4(self):
        """Get an instance of Api Networkv4 services facade."""

        return ApiNetworkIPv4(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)