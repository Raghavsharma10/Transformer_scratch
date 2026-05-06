def create_api_interface_request(self):
        """Get an instance of Api Vip Requests services facade."""

        return ApiInterfaceRequest(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)