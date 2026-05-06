def _init_typedef(self, typedef_curr, name, lnum):
        """Initialize new typedef and perform checks."""
        if typedef_curr is None:
            return TypeDef()
        msg = "PREVIOUS {REC} WAS NOT TERMINATED AS EXPECTED".format(REC=name)
        self._die(msg, lnum)