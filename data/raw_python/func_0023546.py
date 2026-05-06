def do_run(self, arg):
        """run [args...]
        Restart the debugged python program. If a string is supplied
        it is splitted with "shlex", and the result is used as the new
        sys.argv.  History, breakpoints, actions and debugger options
        are preserved.  "restart" is an alias for "run".
        """
        if arg:
            argv0 = sys.argv[0:1]
            sys.argv = shlex.split(arg)
            sys.argv[:0] = argv0
        # this is caught in the main debugger loop
        raise Restart