def setportprio(self, port, prio):
        """ Set port priority value. """
        _runshell([brctlexe, 'setportprio', self.name, port, str(prio)],
            "Could not set priority in port %s in %s." % (port, self.name))