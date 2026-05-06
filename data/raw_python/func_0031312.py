def setmaxage(self, time):
        """ Set bridge max message age time. """
        _runshell([brctlexe, 'setmaxage', self.name, str(time)],
            "Could not set max message age in %s." % self.name)