def create_api_v4_ipv4(self):
        """Get an instance of Api V4 IPv4 services facade."""

        return ApiV4IPv4(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)