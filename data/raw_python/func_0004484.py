def create_api_option_vip(self):
        """Get an instance of Api Option Vip services facade."""
        return ApiOptionVip(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)