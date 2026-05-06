def create_api_ipv4(self):
        """Get an instance of Api IPv4 services facade."""

        return ApiIPv4(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)