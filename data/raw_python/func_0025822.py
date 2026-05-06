def hasUnsavedChanges(self):
        """ Determine if there are any edits in the GUI that have not yet been
        saved (e.g. to a file). """

        # Sanity check - this case shouldn't occur
        assert self._lastSavedState is not None, \
               "BUG: Please report this as it should never occur."

        # Force the current GUI values into our model in memory, but don't
        # change anything.  Don't save to file, don't even convert bad
        # values to their previous state in the gui.  Note that this can
        # leave the GUI in a half-saved state, but since we are about to exit
        # this is OK.  We only want prompting to occur if they decide to save.
        badList = self.checkSetSaveEntries(doSave=False, fleeOnBadVals=True,
                                           allowGuiChanges=False)
        if badList:
            return True

        # Then compare our data to the last known saved state.  MAKE SURE
        # the LHS is the actual dict (and not 'self') to invoke the dict
        # comparison only.
        return self._lastSavedState != self._taskParsObj