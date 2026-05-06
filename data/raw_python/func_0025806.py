def setAllEntriesFromParList(self, aParList, updateModel=False):
        """ Set all the parameter entry values in the GUI to the values
            in the given par list. If 'updateModel' is True, the internal
            param list will be updated to the new values as well as the GUI
            entries (slower and not always necessary). Note the
            corresponding TparDisplay method. """

        # Get model data, the list of pars
        theParamList = self._taskParsObj.getParList() # we may modify members

        if len(aParList) != len(theParamList):
            showwarning(message="Attempting to set parameter values from a "+ \
                        "list of different length ("+str(len(aParList))+ \
                        ") than the number shown here ("+ \
                        str(len(theParamList))+").  Be aware.",
                        title="Parameter List Length Mismatch")

        # LOOP THRU GUI PAR LIST
        for i in range(self.numParams):
            par = theParamList[i]
            if par.type == "pset":
                continue # skip PSET's for now
            gui_entry = self.entryNo[i]

            # Set the value in the paramList before setting it in the GUI
            # This may be in the form of a list, or an IrafParList (getValue)
            if isinstance(aParList, list):
                # Since "aParList" can have them in different order and number
                # than we do, we'll have to first find the matching param.
                found = False
                for newpar in aParList:
                    if newpar.name==par.name and newpar.scope==par.scope:
                        par.set(newpar.value) # same as .get(native=1,prompt=0)
                        found = True
                        break

                # Now see if newpar was found in our list
                if not found:
                    pnm = par.name
                    if len(par.scope): pnm = par.scope+'.'+par.name
                    raise UnfoundParamError('Error - Unfound Parameter! \n\n'+\
                      'Expected parameter "'+pnm+'" for task "'+ \
                      self.taskName+'". \nThere may be others...')

            else: # assume has getValue()
                par.set(aParList.getValue(par.name, native=1, prompt=0))

            # gui holds a str, but par.value is native; conversion occurs
            gui_entry.forceValue(par.value, noteEdited=False) # no triggers yet

        if updateModel:
            # Update the model values via checkSetSaveEntries
            self.badEntriesList = self.checkSetSaveEntries(doSave=False)

            # If there were invalid entries, prepare the message dialog
            if self.badEntriesList:
                self.processBadEntries(self.badEntriesList,
                                       self.taskName, canCancel=False)