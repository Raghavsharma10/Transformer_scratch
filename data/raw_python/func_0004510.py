def create_ip(self):
        """Get an instance of ip services facade."""
        return Ip(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)