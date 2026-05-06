def _handleParListMismatch(self, probStr, extra=False):
        """ Handle the situation where two par lists do not match.
        This is meant to allow subclasses to override. Note that this only
        handles "missing" pars and "extra" pars, not wrong-type pars. """

        errmsg = 'ERROR: mismatch between default and current par lists ' + \
               'for task "'+self.taskName+'"'
        if probStr:
            errmsg += '\n\t'+probStr
        errmsg += '\n(try: "unlearn '+self.taskName+'")'
        print(errmsg)
        return False