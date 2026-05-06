def create_environment(self, name, default=False, zone=None):
        """ Creates environment and returns Environment object.
        """
        from qubell.api.private.environment import Environment
        return Environment.new(organization=self, name=name, zone_id=zone, default=default, router=self._router)