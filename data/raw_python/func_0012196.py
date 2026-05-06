def read(self):
        '''Execute the expression and capture its output, similar to backticks
        or $() in the shell. This is a wrapper around run() which captures
        stdout, decodes it, trims it, and returns it directly.'''
        result = self.stdout_capture().run()
        stdout_str = decode_with_universal_newlines(result.stdout)
        return stdout_str.rstrip('\n')