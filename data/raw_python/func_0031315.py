def _show(self):
        """ Return a list of unsorted bridge details. """ 
        p = _runshell([brctlexe, 'show', self.name],
            "Could not show %s." % self.name)
        return p.stdout.read().split()[7:]