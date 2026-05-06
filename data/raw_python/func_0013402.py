def _init_goterm_ref(self, rec_curr, name, lnum):
        """Initialize new reference and perform checks."""
        if rec_curr is None:
            return GOTerm()
        msg = "PREVIOUS {REC} WAS NOT TERMINATED AS EXPECTED".format(REC=name)
        self._die(msg, lnum)