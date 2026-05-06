def create_interface(self):
        """Get an instance of interface services facade."""
        return Interface(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)