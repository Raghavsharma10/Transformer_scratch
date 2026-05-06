def pfopen(self, event=None):
        """ Load the parameter settings from a user-specified file. """

        # Get the selected file name
        fname = self._openMenuChoice.get()

        # Also allow them to simply find any file - do not check _task_name_...
        # (could use tkinter's FileDialog, but this one is prettier)
        if fname[-3:] == '...':
            if capable.OF_TKFD_IN_EPAR:
                fname = askopenfilename(title="Load Config File",
                                        parent=self.top)
            else:
                from . import filedlg
                fd = filedlg.PersistLoadFileDialog(self.top,
                                                   "Load Config File",
                                                   self._getSaveAsFilter())
                if fd.Show() != 1:
                    fd.DialogCleanup()
                    return
                fname = fd.GetFileName()
                fd.DialogCleanup()

        if not fname: return # canceled
        self.debug('Loading from: '+fname)

        # load it into a tmp object (use associatedPkg if we have one)
        try:
            tmpObj = cfgpars.ConfigObjPars(fname, associatedPkg=\
                                           self._taskParsObj.getAssocPkg(),
                                           strict=self._strict)
        except Exception as ex:
            showerror(message=ex.message, title='Error in '+os.path.basename(fname))
            self.debug('Error in '+os.path.basename(fname))
            self.debug(traceback.format_exc())
            return

        # check it to make sure it is a match
        if not self._taskParsObj.isSameTaskAs(tmpObj):
            msg = 'The current task is "'+self._taskParsObj.getName()+ \
                  '", but the selected file is for task "'+ \
                  str(tmpObj.getName())+'".  This file was not loaded.'
            showerror(message=msg, title="Error in "+os.path.basename(fname))
            self.debug(msg)
            self.debug(traceback.format_exc())
            return

        # Set the GUI entries to these values (let the user Save after)
        newParList = tmpObj.getParList()
        try:
            self.setAllEntriesFromParList(newParList, updateModel=True)
                # go ahead and updateModel, even though it will take longer,
                # we need it updated for the copy of the dict we make below
        except editpar.UnfoundParamError as pe:
            showwarning(message=str(pe), title="Error in "+os.path.basename(fname))
        # trip any triggers
        self.checkAllTriggers('fopen')

        # This new fname is our current context
        self.updateTitle(fname)
        self._taskParsObj.filename = fname # !! maybe try setCurrentContext() ?
        self.freshenFocus()
        self.showStatus("Loaded values from: "+fname, keep=2)

        # Since we are in a new context (and have made no changes yet), make
        # a copy so we know what the last state was.
        # The dict() method returns a deep-copy dict of the keyvals.
        self._lastSavedState = self._taskParsObj.dict()