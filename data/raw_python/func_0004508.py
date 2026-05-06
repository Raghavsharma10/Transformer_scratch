def create_healthcheck(self):
        """Get an instance of Poll services facade."""

        return Healthcheck(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)