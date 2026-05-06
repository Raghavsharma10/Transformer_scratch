def do_debug(self, arg):
        """debug code
        Enter a recursive debugger that steps through the code
        argument (which is an arbitrary expression or statement to be
        executed in the current environment).
        """
        self.settrace(False)
        globals = self.curframe.f_globals
        locals = self.get_locals(self.curframe)
        p = Pdb(self.completekey, self.stdin, self.stdout, debug=True)
        p.prompt = "(%s) " % self.prompt.strip()
        self.message("ENTERING RECURSIVE DEBUGGER")
        sys.call_tracing(p.run, (arg, globals, locals))
        self.message("LEAVING RECURSIVE DEBUGGER")
        self.settrace(True)
        self.lastcmd = p.lastcmd