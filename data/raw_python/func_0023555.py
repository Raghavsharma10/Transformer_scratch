def do_longlist(self, arg):
        """longlist | ll
        List the whole source code for the current function or frame.
        """
        filename = self.curframe.f_code.co_filename
        breaklist = self.get_file_breaks(filename)
        try:
            lines, lineno = getsourcelines(self.curframe,
                                self.get_locals(self.curframe))
        except IOError as err:
            self.error(err)
            return
        self._print_lines(lines, lineno, breaklist, self.curframe)