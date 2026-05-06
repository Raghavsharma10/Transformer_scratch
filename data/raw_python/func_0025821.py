def _doActualSave(self, fname, comment, set_ro=False, overwriteRO=False):
        """ Override this so we can handle case of file not writable, as
            well as to make our _lastSavedState copy. """
        self.debug('Saving, file name given: '+str(fname)+', set_ro: '+\
                   str(set_ro)+', overwriteRO: '+str(overwriteRO))
        cantWrite = False
        inInstArea = False
        if fname in (None, ''): fname = self._taskParsObj.getFilename()
        # now do some final checks then save
        try:
            if _isInstalled(fname): # check: may be installed but not read-only
                inInstArea = cantWrite = True
            else:
                # in case of save-as, allow overwrite of read-only file
                if overwriteRO and os.path.exists(fname):
                    setWritePrivs(fname, True, True) # try make writable
                # do the save
                rv=self._taskParsObj.saveParList(filename=fname,comment=comment)
        except IOError:
            cantWrite = True

        # User does not have privs to write to this file. Get name of local
        # choice and try to use that.
        if cantWrite:
            fname = self._taskParsObj.getDefaultSaveFilename()
            # Tell them the context is changing, and where we are saving
            msg = 'Read-only config file for task "'
            if inInstArea:
                msg = 'Installed config file for task "'
            msg += self._taskParsObj.getName()+'" is not to be overwritten.'+\
                  '  Values will be saved to: \n\n\t"'+fname+'".'
            showwarning(message=msg, title="Will not overwrite!")
            # Try saving to their local copy
            rv=self._taskParsObj.saveParList(filename=fname, comment=comment)

        # Treat like a save-as (update title for ALL save ops)
        self._saveAsPostSave_Hook(fname)

        # Limit write privs if requested (only if not in the rc dir)
        if set_ro and os.path.dirname(os.path.abspath(fname)) != \
                                      os.path.abspath(self._rcDir):
            cfgpars.checkSetReadOnly(fname)

        # Before returning, make a copy so we know what was last saved.
        # The dict() method returns a deep-copy dict of the keyvals.
        self._lastSavedState = self._taskParsObj.dict()
        return rv