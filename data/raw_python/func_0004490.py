def create_api_object_group_permission(self):
        """Get an instance of Api Vip Requests services facade."""

        return ApiObjectGroupPermission(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)