def create_api_environment_vip(self):
        """Get an instance of Api Environment Vip services facade."""
        return ApiEnvironmentVip(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)