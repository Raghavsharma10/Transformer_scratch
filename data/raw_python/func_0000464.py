def get_zone(self, id=None, name=None):
        """ Get zone object by name or id.
        """
        log.info("Picking zone: %s (%s)" % (name, id))
        return self.zones[id or name]