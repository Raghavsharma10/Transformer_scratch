def do_continue(self, args):
        """Continue the interpreter
        """
        self._do_print_from_last_cmd = True
        self._interp.cont()
        return True