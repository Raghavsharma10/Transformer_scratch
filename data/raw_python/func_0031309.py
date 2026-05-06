def setbridgeprio(self, prio):
        """ Set bridge priority value. """
        _runshell([brctlexe, 'setbridgeprio', self.name, str(prio)],
            "Could not set bridge priority in %s." % self.name)