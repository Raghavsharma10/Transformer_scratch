def setageing(self, time):
        """ Set bridge ageing time. """
        _runshell([brctlexe, 'setageing', self.name, str(time)],
            "Could not set ageing time in %s." % self.name)