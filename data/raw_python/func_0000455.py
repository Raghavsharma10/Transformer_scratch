def create_instance(self, application, revision=None, environment=None, name=None, parameters=None, submodules=None,
                        destroyInterval=None, manifestVersion=None):
        """ Launches instance in application and returns Instance object.
        """
        from qubell.api.private.instance import Instance
        return Instance.new(self._router, application, revision, environment, name,
                            parameters, submodules, destroyInterval, manifestVersion=manifestVersion)