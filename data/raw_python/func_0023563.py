def do_unalias(self, arg):
        """unalias name
        Delete the specified alias.
        """
        args = arg.split()
        if len(args) == 0: return
        if args[0] in self.aliases:
            del self.aliases[args[0]]