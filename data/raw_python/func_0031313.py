def setpathcost(self, port, cost):
        """ Set port path cost value for STP protocol. """
        _runshell([brctlexe, 'setpathcost', self.name, port, str(cost)],
            "Could not set path cost in port %s in %s." % (port, self.name))