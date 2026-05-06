def create_api_vip_request(self):
        """Get an instance of Api Vip Requests services facade."""

        return ApiVipRequest(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap,
            self.log_level)