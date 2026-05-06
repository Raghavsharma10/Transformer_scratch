def _die(self, msg, lnum):
        """Raise an Exception if file read is unexpected."""
        raise Exception("**FATAL {FILE}({LNUM}): {MSG}\n".format(
            FILE=self.obo_file, LNUM=lnum, MSG=msg))