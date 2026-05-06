def setfd(self, time):
        """ Set bridge forward delay time value. """
        _runshell([brctlexe, 'setfd', self.name, str(time)],
            "Could not set forward delay in %s." % self.name)