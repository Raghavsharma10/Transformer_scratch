def do_help(self, arg):
        """h(elp)
        Without argument, print the list of available commands.
        With a command name as argument, print help about that command.
        "help pdb" shows the full pdb documentation.
        "help exec" gives help on the ! command.
        """
        if not arg:
            return cmd.Cmd.do_help(self, arg)
        try:
            try:
                topic = getattr(self, 'help_' + arg)
                return topic()
            except AttributeError:
                command = getattr(self, 'do_' + arg)
        except AttributeError:
            self.error('No help for %r' % arg)
        else:
            if sys.flags.optimize >= 2:
                self.error('No help for %r; please do not run Python with -OO '
                           'if you need command help' % arg)
                return
            self.message(command.__doc__.rstrip())