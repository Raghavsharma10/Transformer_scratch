def hairpin(self, port, val=True):
        """ Turn harpin on/off on a port. """ 
        if val: state = 'on' 
        else: state = 'off'
        _runshell([brctlexe, 'hairpin', self.name, port, state],
            "Could not set hairpin in port %s in %s." % (port, self.name))