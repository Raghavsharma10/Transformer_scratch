def get_environment(self, id=None, name=None):
        """ Get environment object by name or id.
        """
        log.info("Picking environment: %s (%s)" % (name, id))
        return self.environments[id or name]