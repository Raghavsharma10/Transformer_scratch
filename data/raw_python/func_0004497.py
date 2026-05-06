def create_environment_vip(self):
        """Get an instance of environment_vip services facade."""
        return EnvironmentVIP(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)