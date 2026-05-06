def stp(self, val=True):
        """ Turn STP protocol on/off. """
        if val: state = 'on' 
        else: state = 'off'
        _runshell([brctlexe, 'stp', self.name, state],
            "Could not set stp on %s." % self.name)