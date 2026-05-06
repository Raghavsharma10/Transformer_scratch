def edited(self, scope, name, lastSavedVal, newVal, action):
        """ This is the callback function invoked when an item is edited.
            This is only called for those items which were previously
            specified to use this mechanism.  We do not turn this on for
            all items because the performance might be prohibitive.
            This kicks off any previously registered triggers. """

        # Get name(s) of any triggers that this par triggers
        triggerNamesTup = self._taskParsObj.getTriggerStrings(scope, name)
        assert triggerNamesTup is not None and len(triggerNamesTup) > 0, \
               'Empty trigger name for: "'+name+'", consult the .cfgspc file.'

        # Loop through all trigger names - each one is a trigger to kick off -
        # in the order that they appear in the tuple we got.  Most cases will
        # probably only have a single trigger in the tuple.
        for triggerName in triggerNamesTup:
            # First handle the known/canned trigger names
#           print (scope, name, newVal, action, triggerName) # DBG: debug line

            # _section_switch_
            if triggerName == '_section_switch_':
                # Try to uniformly handle all possible par types here, not
                # just boolean (e.g. str, int, float, etc.)
                # Also, see logic in _BooleanMixin._coerceOneValue()
                state = newVal not in self.FALSEVALS
                self._toggleSectionActiveState(scope, state, (name,))
                continue

            # _2_section_switch_ (see notes above in _section_switch_)
            if triggerName == '_2_section_switch_':
                state = newVal not in self.FALSEVALS
                # toggle most of 1st section (as usual) and ALL of next section
                self._toggleSectionActiveState(scope, state, (name,))
                # get first par of next section (fpons) - is a tuple
                fpons = self.findNextSection(scope, name)
                nextSectScope = fpons[0]
                if nextSectScope:
                    self._toggleSectionActiveState(nextSectScope, state, None)
                continue

            # Now handle rules with embedded code (eg. triggerName=='_rule1_')
            if '_RULES_' in self._taskParsObj and \
               triggerName in self._taskParsObj['_RULES_'].configspec:
                # Get codeStr to execute it, but before we do so, check 'when' -
                # make sure this is an action that is allowed to cause a trigger
                ruleSig = self._taskParsObj['_RULES_'].configspec[triggerName]
                chkArgsDict = vtor_checks.sigStrToKwArgsDict(ruleSig)
                codeStr = chkArgsDict.get('code') # or None if didn't specify
                when2run = chkArgsDict.get('when') # or None if didn't specify

                greenlight = False # do we have a green light to eval the rule?
                if when2run is None:
                    greenlight = True # means run rule for any possible action
                else: # 'when' was set to something so we need to check action
                    # check value of action (poor man's enum)
                    assert action in editpar.GROUP_ACTIONS, \
                        "Unknown action: "+str(action)+', expected one of: '+ \
                        str(editpar.GROUP_ACTIONS)
                    # check value of 'when' (allow them to use comma-sep'd str)
                    # (readers be aware that values must be those possible for
                    #  'action', and 'always' is also allowed)
                    whenlist = when2run.split(',')
                    # warn for invalid values
                    for w in whenlist:
                        if not w in editpar.GROUP_ACTIONS and w != 'always':
                           print('WARNING: skipping bad value for when kwd: "'+\
                                  w+'" in trigger/rule: '+triggerName)
                    # finally, do the correlation
                    greenlight = 'always' in whenlist or action in whenlist

                # SECURITY NOTE: because this part executes arbitrary code, that
                # code string must always be found only in the configspec file,
                # which is intended to only ever be root-installed w/ the pkg.
                if codeStr:
                    if not greenlight:
                        continue # not an error, just skip this one
                    self.showStatus("Evaluating "+triggerName+' ...') #dont keep
                    self.top.update_idletasks() #allow msg to draw prior to exec
                    # execute it and retrieve the outcome
                    try:
                        outval = execEmbCode(scope, name, newVal, self, codeStr)
                    except Exception as ex:
                        outval = 'ERROR in '+triggerName+': '+str(ex)
                        print(outval)
                        msg = outval+':\n'+('-'*99)+'\n'+traceback.format_exc()
                        msg += 'CODE:  '+codeStr+'\n'+'-'*99+'\n'
                        self.debug(msg)
                        self.showStatus(outval, keep=1)

                    # Leave this debug line in until it annoys someone
                    msg = 'Value of "'+name+'" triggered "'+triggerName+'"'
                    stroutval = str(outval)
                    if len(stroutval) < 30: msg += '  -->  "'+stroutval+'"'
                    self.showStatus(msg, keep=0)
                    # Now that we have triggerName evaluated to outval, we need
                    # to look through all the parameters and see if there are
                    # any items to be affected by triggerName (e.g. '_rule1_')
                    self._applyTriggerValue(triggerName, outval)
                    continue

            # If we get here, we have an unknown/unusable trigger
            raise RuntimeError('Unknown trigger for: "'+name+'", named: "'+ \
                  str(triggerName)+'".  Please consult the .cfgspc file.')