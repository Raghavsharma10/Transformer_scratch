def _toggleSectionActiveState(self, sectionName, state, skipList):
        """ Make an entire section (minus skipList items) either active or
            inactive.  sectionName is the same as the param's scope. """

        # Get model data, the list of pars
        theParamList = self._taskParsObj.getParList()

        # Loop over their assoc. entries
        for i in range(self.numParams):
            if theParamList[i].scope == sectionName:
                if skipList and theParamList[i].name in skipList:
#                   self.entryNo[i].setActiveState(True) # these always active
                    pass # if it started active, we don't need to reactivate it
                else:
                    self.entryNo[i].setActiveState(state)