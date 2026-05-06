def do_quit(self, args):
        """The quit command
        """
        self._interp.set_break(self._interp.BREAK_NONE)
        return True