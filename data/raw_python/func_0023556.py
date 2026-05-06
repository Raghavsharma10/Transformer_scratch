def do_source(self, arg):
        """source expression
        Try to get source code for the given object and display it.
        """
        try:
            obj = self._getval(arg)
        except Exception:
            return
        try:
            lines, lineno = getsourcelines(obj, self.get_locals(self.curframe))
        except (IOError, TypeError) as err:
            self.error(err)
            return
        self._print_lines(lines, lineno)