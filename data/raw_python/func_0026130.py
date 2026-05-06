def flagThisPar(self, currentVal, force):
        """ If this par's value is different from the default value, it is here
        that we flag it somehow as such.  This basic version simply makes the
        surrounding text red (or returns it to normal). May be overridden.
        Leave force at False if you want to allow this mehtod to make smart
        time-saving decisions about when it can skip recoloring because it is
        already the right color. Set force to true if you think we got out
        of sync and need to be fixed. """

        # Get out ASAP if we can
        if (not force) and (not self._flagNonDefaultVals): return

        # handle simple case before comparing values (quick return)
        if force and not self._flagNonDefaultVals:
            self._flagged = False
            self.promptLabel.configure(fg="black")
            return

        # Get/format values to compare
        currentNative = self.convertToNative(currentVal)
        defaultNative = self.convertToNative(self.defaultParamInfo.value)
        # par.value is same as par.get(native=1,prompt=0)

        # flag or unflag as needed
        if currentNative != defaultNative:
            if not self._flagged or force:
                self._flagged = True
                self.promptLabel.configure(fg=self._flaggedColor) # was red
        else: # same as def
            if self._flagged or force:
                self._flagged = False
                self.promptLabel.configure(fg="black")