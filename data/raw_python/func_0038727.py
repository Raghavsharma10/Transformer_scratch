def log(self, logfile=None):
        """Log the ASCII traceback into a file object."""
        if logfile is None:
            logfile = sys.stderr
        tb = self.plaintext.rstrip() + '\n'

        file_mode = getattr(logfile, 'mode', None)
        if file_mode is not None:
            if 'b' in file_mode:
                tb = tb.encode('utf-8')
            logfile.write(tb)