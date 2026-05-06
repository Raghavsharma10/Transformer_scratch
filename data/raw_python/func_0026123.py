def widgetEdited(self, event=None, val=None, action='entry', skipDups=True):
        """ A general method for firing any applicable triggers when
            a value has been set.  This is meant to be easily callable from any
            part of this class (or its subclasses), so that it can be called
            as soon as need be (immed. on click?).  This is smart enough to
            be called multiple times, itself handling the removal of any/all
            duplicate successive calls (unless skipDups is False). If val is
            None, it will use the GUI entry's current value via choice.get().
            See teal.py for a description of action.
        """

        # be as lightweight as possible if obj doesn't care about this stuff
        if not self._editedCallbackObj and not self._flagNonDefaultVals:
            return

        # get the current value
        curVal = val # take this first, if it is given
        if curVal is None:
            curVal = self.choice.get()

        # do any flagging
        self.flagThisPar(curVal, False)

        # see if this is a duplicate successive call for the same value
        if skipDups and curVal==self._lastWidgetEditedVal: return

        # pull trigger
        if not self._editedCallbackObj: return
        self._editedCallbackObj.edited(self.paramInfo.scope,
                                       self.paramInfo.name,
                                       self.previousValue, curVal,
                                       action)
        # for our duplicate checker
        self._lastWidgetEditedVal = curVal