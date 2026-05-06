def create_api_v4_neighbor(self):
        """Get an instance of Api Neighbor services facade."""
        return ApiV4Neighbor(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)