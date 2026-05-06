def create_api_v4_equipment(self):
        """Get an instance of Api Equipment services facade."""
        return ApiV4Equipment(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)