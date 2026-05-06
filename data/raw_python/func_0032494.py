def _prepareSubForm(self, liveForm):
        """
        Utility for turning liveforms into subforms, and compacting them as
        necessary.

        @param liveForm: a liveform.
        @type liveForm: L{LiveForm}

        @return: a sub form.
        @rtype: L{LiveForm}
        """
        liveForm = liveForm.asSubForm(self.name) # XXX Why did this work???
        # if we are compact, tell the liveform so it can tell its parameters
        # also
        if self._parameterIsCompact:
            liveForm.compact()
        return liveForm