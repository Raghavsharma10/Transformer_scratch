def create_pool(self):
        """Get an instance of Poll services facade."""

        return Pool(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)