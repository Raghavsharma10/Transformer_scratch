def create_api_ipv6(self):
        """Get an instance of Api IPv6 services facade."""

        return ApiIPv6(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)