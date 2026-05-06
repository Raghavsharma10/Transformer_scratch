def create_rack(self):
        """Get an instance of rack services facade."""
        return Rack(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)