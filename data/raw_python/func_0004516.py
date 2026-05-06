def create_option_pool(self):
        """Get an instance of option_pool services facade."""
        return OptionPool(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)