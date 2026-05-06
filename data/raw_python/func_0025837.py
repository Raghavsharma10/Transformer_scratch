def _applyTriggerValue(self, triggerName, outval):
        """ Here we look through the entire .cfgspc to see if any parameters
        are affected by this trigger. For those that are, we apply the action
        to the GUI widget.  The action is specified by depType. """

        # First find which items are dependent upon this trigger (cached)
        # e.g. { scope1.name1 : dep'cy-type, scope2.name2 : dep'cy-type, ... }
        depParsDict = self._taskParsObj.getParsWhoDependOn(triggerName)
        if not depParsDict: return
        if 0: print("Dependent parameters:\n"+str(depParsDict)+"\n")

        # Get model data, the list of pars
        theParamList = self._taskParsObj.getParList()

        # Then go through the dependent pars and apply the trigger to them
        settingMsg = ''
        for absName in depParsDict:
            used = False
            # For each dep par, loop to find the widget for that scope.name
            for i in range(self.numParams):
                scopedName = theParamList[i].scope+'.'+theParamList[i].name # diff from makeFullName!!
                if absName == scopedName: # a match was found
                    depType = depParsDict[absName]
                    if depType == 'active_if':
                        self.entryNo[i].setActiveState(outval)
                    elif depType == 'inactive_if':
                        self.entryNo[i].setActiveState(not outval)
                    elif depType == 'is_set_by':
                        self.entryNo[i].forceValue(outval, noteEdited=True)
                        # WARNING! noteEdited=True may start recursion!
                        if len(settingMsg) > 0: settingMsg += ", "
                        settingMsg += '"'+theParamList[i].name+'" to "'+\
                                      outval+'"'
                    elif depType in ('set_yes_if', 'set_no_if'):
                        if bool(outval):
                            newval = 'yes'
                            if depType == 'set_no_if': newval = 'no'
                            self.entryNo[i].forceValue(newval, noteEdited=True)
                            # WARNING! noteEdited=True may start recursion!
                            if len(settingMsg) > 0: settingMsg += ", "
                            settingMsg += '"'+theParamList[i].name+'" to "'+\
                                          newval+'"'
                        else:
                            if len(settingMsg) > 0: settingMsg += ", "
                            settingMsg += '"'+theParamList[i].name+\
                                          '" (no change)'
                    elif depType == 'is_disabled_by':
                        # this one is only used with boolean types
                        on = self.entryNo[i].convertToNative(outval)
                        if on:
                            # do not activate whole section or change
                            # any values, only activate this one
                            self.entryNo[i].setActiveState(True)
                        else:
                            # for off, set the bool par AND grey WHOLE section
                            self.entryNo[i].forceValue(outval, noteEdited=True)
                            self.entryNo[i].setActiveState(False)
                            # we'd need this if the par had no _section_switch_
#                           self._toggleSectionActiveState(
#                                theParamList[i].scope, False, None)
                            if len(settingMsg) > 0: settingMsg += ", "
                            settingMsg += '"'+theParamList[i].name+'" to "'+\
                                          outval+'"'
                    else:
                        raise RuntimeError('Unknown dependency: "'+depType+ \
                                           '" for par: "'+scopedName+'"')
                    used = True
                    break

            # Or maybe it is a whole section
            if absName.endswith('._section_'):
                scope = absName[:-10]
                depType = depParsDict[absName]
                if depType == 'active_if':
                    self._toggleSectionActiveState(scope, outval, None)
                elif depType == 'inactive_if':
                    self._toggleSectionActiveState(scope, not outval, None)
                used = True

            # Help to debug the .cfgspc rules
            if not used:
                raise RuntimeError('UNUSED "'+triggerName+'" dependency: '+ \
                      str({absName:depParsDict[absName]}))

        if len(settingMsg) > 0:
# why ?!    self.freshenFocus()
            self.showStatus('Automatically set '+settingMsg, keep=1)