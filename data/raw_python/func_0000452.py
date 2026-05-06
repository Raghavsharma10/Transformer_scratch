def get_application(self, id=None, name=None):
        """ Get application object by name or id.
        """
        log.info("Picking application: %s (%s)" % (name, id))
        return self.applications[id or name]