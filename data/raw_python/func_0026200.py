def reset(self):
        """Clear ConfigObj instance and restore to 'freshly created' state."""
        self.clear()
        self._initialise()
        # FIXME: Should be done by '_initialise', but ConfigObj constructor (and reload)
        #        requires an empty dictionary
        self.configspec = None
        # Just to be sure ;-)
        self._original_configspec = None