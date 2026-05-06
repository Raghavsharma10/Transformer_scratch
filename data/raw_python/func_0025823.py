def _defineEditedCallbackObjectFor(self, parScope, parName):
        """ Override to allow us to use an edited callback. """

        # We know that the _taskParsObj is a ConfigObjPars
        triggerStrs = self._taskParsObj.getTriggerStrings(parScope, parName)

        # Some items will have a trigger, but likely most won't
        if triggerStrs and len(triggerStrs) > 0:
            return self
        else:
            return None