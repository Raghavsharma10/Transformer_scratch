def create_api_pool_deploy(self):
        """Get an instance of Api Pool Deploy services facade."""
        return ApiPoolDeploy(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)