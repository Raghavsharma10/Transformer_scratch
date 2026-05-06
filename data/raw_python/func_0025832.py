def _setToDefaults(self):
        """ Load the default parameter settings into the GUI. """

        # Create an empty object, where every item is set to it's default value
        try:
            tmpObj = cfgpars.ConfigObjPars(self._taskParsObj.filename,
                                           associatedPkg=\
                                           self._taskParsObj.getAssocPkg(),
                                           setAllToDefaults=self.taskName,
                                           strict=False)
        except Exception as ex:
            msg = "Error Determining Defaults"
            showerror(message=msg+'\n\n'+ex.message, title="Error Determining Defaults")
            return

        # Set the GUI entries to these values (let the user Save after)
        tmpObj.filename = self._taskParsObj.filename = '' # name it later
        newParList = tmpObj.getParList()
        try:
            self.setAllEntriesFromParList(newParList) # needn't updateModel yet
            self.checkAllTriggers('defaults')
            self.updateTitle('')
            self.showStatus("Loaded default "+self.taskName+" values via: "+ \
                 os.path.basename(tmpObj._original_configspec), keep=1)
        except editpar.UnfoundParamError as pe:
            showerror(message=str(pe), title="Error Setting to Default Values")