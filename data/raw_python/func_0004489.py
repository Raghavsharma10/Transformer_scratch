def create_api_object_type(self):
        """Get an instance of Api Vip Requests services facade."""

        return ApiObjectType(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)