def delif(self, iname):
        """ Delete an interface from the bridge. """
        _runshell([brctlexe, 'delif', self.name, iname],
            "Could not delete interface %s from %s." % (iname, self.name))