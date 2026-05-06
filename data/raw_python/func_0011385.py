def do_next(self, args):
        """Step over the next statement
        """
        self._do_print_from_last_cmd = True
        self._interp.step_over()
        return True