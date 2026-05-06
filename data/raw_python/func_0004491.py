def create_api_object_group_permission_general(self):
        """Get an instance of Api Vip Requests services facade."""

        return ApiObjectGroupPermissionGeneral(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)