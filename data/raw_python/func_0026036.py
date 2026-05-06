def _optionalPrompt(self, mode):
        """Interactively prompt for parameter if necessary

        Prompt for value if
        (1) mode is hidden but value is undefined or bad, or
        (2) mode is query and value was not set on command line
        Never prompt for "u" mode parameters, which are local variables.
        """
        if (self.mode == "h") or (self.mode == "a" and mode == "h"):
            # hidden parameter
            if not self.isLegal():
                self.getWithPrompt()
        elif self.mode == "u":
            # "u" is a special mode used for local variables in CL scripts
            # They should never prompt under any circumstances
            if not self.isLegal():
                raise ValueError(
                                "Attempt to access undefined local variable `%s'" %
                                self.name)
        else:
            # query parameter
            if self.isCmdline()==0:
                self.getWithPrompt()