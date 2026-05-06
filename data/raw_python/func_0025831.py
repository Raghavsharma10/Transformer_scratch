def _handleParListMismatch(self, probStr, extra=False):
        """ Override to include ConfigObj filename and specific errors.
        Note that this only handles "missing" pars and "extra" pars, not
        wrong-type pars.  So it isn't that big of a deal. """

        # keep down the duplicate errors
        if extra:
            return True # the base class is already stating it will be ignored

        # find the actual errors, and then add that to the generic message
        errmsg = 'Warning: '
        if self._strict:
            errmsg = 'ERROR: '
        errmsg = errmsg+'mismatch between default and current par lists ' + \
                 'for task "'+self.taskName+'".'
        if probStr:
            errmsg += '\n\t'+probStr
        errmsg += '\nTry editing/deleting: "' + \
                  self._taskParsObj.filename+'" (or, if in PyRAF: "unlearn ' + \
                  self.taskName+'").'
        print(errmsg)
        return True