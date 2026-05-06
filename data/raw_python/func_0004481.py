def create_api_v4_ipv6(self):
        """Get an instance of Api V4 IPv6 services facade."""

        return ApiV4IPv6(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)