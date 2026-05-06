def get_role(self, id=None, name=None):
        """ Get role object by name or id.
        """
        log.info("Picking role: %s (%s)" % (name, id))
        return self.roles[id or name]