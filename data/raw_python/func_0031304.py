def addif(self, iname):
        """ Add an interface to the bridge """
        _runshell([brctlexe, 'addif', self.name, iname],
            "Could not add interface %s to %s." % (iname, self.name))