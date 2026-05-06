def get_or_create_instance(self, id=None, application=None, revision=None, environment=None, name=None, parameters=None, submodules=None,
                               destroyInterval=None):
        """ Get instance by id or name.
        If not found: create with given parameters
        """
        try:
            instance = self.get_instance(id=id, name=name)
            if name and name != instance.name:
                instance.rename(name)
                instance.ready()
            return instance
        except exceptions.NotFoundError:
            return self.create_instance(application, revision, environment, name, parameters, submodules, destroyInterval)