def loadDict(self, theDict):
        """ Load the parameter settings from a given dict into the GUI. """
        # We are going to have to merge this info into ourselves so let's
        # first make sure all of our models are up to date with the values in
        # the GUI right now.
        badList = self.checkSetSaveEntries(doSave=False)
        if badList:
            if not self.processBadEntries(badList, self.taskName):
                return
        # now, self._taskParsObj is up-to-date
        # So now we update _taskParsObj with the input dict
        cfgpars.mergeConfigObj(self._taskParsObj, theDict)
        # now sync the _taskParsObj dict with its par list model
        #    '\n'.join([str(jj) for jj in self._taskParsObj.getParList()])
        self._taskParsObj.syncParamList(False)

        # Set the GUI entries to these values (let the user Save after)
        try:
            self.setAllEntriesFromParList(self._taskParsObj.getParList(),
                                          updateModel=True)
            self.checkAllTriggers('fopen')
            self.freshenFocus()
            self.showStatus('Loaded '+str(len(theDict))+ \
                ' user par values for: '+self.taskName, keep=1)
        except Exception as ex:
            showerror(message=ex.message, title="Error Setting to Loaded Values")