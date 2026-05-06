def get_instance(self, id=None, name=None):
        """ Get instance object by name or id.
        If application set, search within the application.
        """
        log.info("Picking instance: %s (%s)" % (name, id))
        if id:  # submodule instances are invisible for lists
            return Instance(id=id, organization=self).init_router(self._router)
        return Instance.get(self._router, self, name)