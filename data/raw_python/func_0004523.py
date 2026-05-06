def create_roteiro(self):
        """Get an instance of roteiro services facade."""
        return Roteiro(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)