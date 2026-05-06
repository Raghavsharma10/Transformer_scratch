def do_interact(self, arg):
        """interact

        Start an interative interpreter whose global namespace
        contains all the (global and local) names found in the current scope.
        """
        def readfunc(prompt):
            self.stdout.write(prompt)
            self.stdout.flush()
            line = self.stdin.readline()
            line = line.rstrip('\r\n')
            if line == 'EOF':
                raise EOFError
            return line

        ns = self.curframe.f_globals.copy()
        ns.update(self.get_locals(self.curframe))
        if isinstance(self.stdin, RemoteSocket):
            # Main interpreter redirection of the code module.
            if PY3:
                import sys as _sys
            else:
                # Parent module 'pdb_clone' not found while handling absolute
                # import.
                _sys = __import__('sys', level=0)
            code.sys = _sys
            self.redirect(code.interact, local=ns, readfunc=readfunc)
        else:
            code.interact("*interactive*", local=ns)