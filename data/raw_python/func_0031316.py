def addbr(self, name):
        """ Create a bridge and set the device up. """
        _runshell([brctlexe, 'addbr', name],
            "Could not create bridge %s." % name)
        _runshell([ipexe, 'link', 'set', 'dev', name, 'up'],
            "Could not set link up for %s." % name)
        return Bridge(name)