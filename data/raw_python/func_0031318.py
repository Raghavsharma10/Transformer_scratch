def showall(self):
        """ Return a list of all available bridges. """
        p = _runshell([brctlexe, 'show'],
            "Could not show bridges.")
        wlist = map(str.split, p.stdout.read().splitlines()[1:])
        brwlist = filter(lambda x: len(x) != 1, wlist)
        brlist = map(lambda x: x[0], brwlist)
        return map(Bridge, brlist)