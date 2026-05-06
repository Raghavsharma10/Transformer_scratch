def create_api_v4_as(self):
        """Get an instance of Api As services facade."""
        return ApiV4As(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)