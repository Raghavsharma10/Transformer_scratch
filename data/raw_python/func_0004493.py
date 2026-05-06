def create_api_vrf(self):
        """Get an instance of Api Vrf services facade."""
        return ApiVrf(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)