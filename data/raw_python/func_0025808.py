def checkSetSaveChildren(self, doSave=True):
        """Check, then set, then save the parameter settings for
        all child (pset) windows.

        Prompts if any problems are found.  Returns None
        on success, list of bad entries on failure.
        """
        if self.isChild:
            return

        # Need to get all the entries and verify them.
        # Save the children in backwards order to coincide with the
        # display of the dialogs (LIFO)
        for n in range (len(self.top.childList)-1, -1, -1):
            self.badEntriesList = self.top.childList[n]. \
                                  checkSetSaveEntries(doSave=doSave)
            if self.badEntriesList:
                ansOKCANCEL = self.processBadEntries(self.badEntriesList,
                              self.top.childList[n].taskName)
                if not ansOKCANCEL:
                    return self.badEntriesList
            # If there were no invalid entries or the user says OK,
            # close down the child and increment to the next child
            self.top.childList[n].top.focus_set()
            self.top.childList[n].top.withdraw()
            del self.top.childList[n]
        # all windows saved successfully
        return