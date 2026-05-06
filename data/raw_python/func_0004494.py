def create_rule(self):
        """Get an instance of block rule services facade."""
        return BlockRule(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)