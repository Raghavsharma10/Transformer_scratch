def delbr(self, name):
        """ Set the device down and delete the bridge. """
        self.getbr(name) # Check if exists
        _runshell([ipexe, 'link', 'set', 'dev', name, 'down'],
            "Could not set link down for %s." % name)
        _runshell([brctlexe, 'delbr', name],
            "Could not delete bridge %s." % name)