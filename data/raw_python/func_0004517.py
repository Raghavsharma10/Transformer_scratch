def create_option_vip(self):
        """Get an instance of option_vip services facade."""
        return OptionVIP(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)