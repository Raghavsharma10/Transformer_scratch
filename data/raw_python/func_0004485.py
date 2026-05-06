def create_api_pool(self):
        """Get an instance of Api Pool services facade."""
        return ApiPool(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)