def create_api_equipment(self):
        """Get an instance of Api Equipment services facade."""
        return ApiEquipment(
            self.networkapi_url,
            self.user,
            self.password,
            self.user_ldap)