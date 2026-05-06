def _print_header(self):
        """Print the header for screen logging"""
        header = " Iter  Dir  "
        if self.constraints is not None:
            header += '  SC CC'
        header += "         Function"
        if self.convergence_condition is not None:
            header += self.convergence_condition.get_header()
        header += "    Time"
        self._screen("-"*(len(header)), newline=True)
        self._screen(header, newline=True)
        self._screen("-"*(len(header)), newline=True)