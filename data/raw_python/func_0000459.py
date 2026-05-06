def instance(self, id=None, application=None, name=None, revision=None, environment=None, parameters=None, submodules=None, destroyInterval=None):
        """ Smart method. It does everything, to return Instance with given parameters within the application.
        If instance found running and given parameters are actual: return it.
        If instance found, but parameters differs - reconfigure instance with new parameters.
        If instance not found: launch instance with given parameters.
        Return: Instance object.
        """
        instance = self.get_or_create_instance(id, application, revision, environment, name, parameters, submodules, destroyInterval)

        reconfigure = False
        # if found:
        #     if revision and revision is not found.revision:
        #         reconfigure = True
        #     if parameters and parameters is not found.parameters:
        #         reconfigure = True

        # We need to reconfigure instance
        if reconfigure:
            instance.reconfigure(revision=revision, parameters=parameters)

        return instance