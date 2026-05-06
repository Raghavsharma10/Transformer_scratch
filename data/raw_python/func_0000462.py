def get_or_create_environment(self, id=None, name=None, zone=None, default=False):
        """ Get environment by id or name.
        If not found: create with given or generated parameters
        """
        if id:
            return self.get_environment(id=id)
        elif name:
            try:
                env = self.get_environment(name=name)
                self._assert_env_and_zone(env, zone)
            except exceptions.NotFoundError:
                env = self.create_environment(name=name, zone=zone, default=default)
            return env
        else:
            name = 'auto-generated-env'
            return self.create_environment(name=name, zone=zone, default=default)