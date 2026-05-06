def getDict(self):
        """ Retrieve the current parameter settings from the GUI."""
        # We are going to have to return the dict so let's
        # first make sure all of our models are up to date with the values in
        # the GUI right now.
        badList = self.checkSetSaveEntries(doSave=False)
        if badList:
            self.processBadEntries(badList, self.taskName, canCancel=False)
        return self._taskParsObj.dict()