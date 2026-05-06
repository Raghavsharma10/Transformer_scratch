def create_vip(self):
        """Get an instance of vip services facade."""
        return Vip(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)