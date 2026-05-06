def create_api_environment(self):
        """Get an instance of Api Environment services facade."""
        return ApiEnvironment(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)