def create_api_vlan(self):
        """Get an instance of Api Vlan services facade."""
        return ApiVlan(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)