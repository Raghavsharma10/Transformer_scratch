def _setTaskParsObj(self, theTask):
        """ Overridden version for ConfigObj. theTask can be either
            a .cfg file name or a ConfigObjPars object. """
        # Create the ConfigObjPars obj
        self._taskParsObj = cfgpars.getObjectFromTaskArg(theTask,
                                    self._strict, False)
        # Tell it that we can be used for catching debug lines
        self._taskParsObj.setDebugLogger(self)

        # Immediately make a copy of it's un-tampered internal dict.
        # The dict() method returns a deep-copy dict of the keyvals.
        self._lastSavedState = self._taskParsObj.dict()