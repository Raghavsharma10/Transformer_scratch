def create_apirack(self):
        """Get an instance of Api Rack Variables services facade."""
        return ApiRack(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)