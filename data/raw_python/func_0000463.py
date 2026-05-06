def environment(self, id=None, name=None, zone=None, default=False):
        """ Smart method. Creates, picks or modifies environment.
        If environment found by name or id parameters not changed: return env.
        If env found by id, but other parameters differs: change them.
        If no environment found, create with given parameters.
        """

        found = False

        # Try to find environment by name or id
        if name and id:
            found = self.get_environment(id=id)
        elif id:
            found = self.get_environment(id=id)
            name = found.name
        elif name:
            try:
                found = self.get_environment(name=name)
            except exceptions.NotFoundError:
                pass

        # If found - compare parameters
        if found:
            self._assert_env_and_zone(found, zone)
            if default and not found.isDefault:
                found.set_as_default()
            # TODO: add abilities to change name.
        if not found:
            created = self.create_environment(name=name, zone=zone, default=default)
        return found or created