def sethello(self, time):
        """ Set bridge hello time value. """
        _runshell([brctlexe, 'sethello', self.name, str(time)],
            "Could not set hello time in %s." % self.name)