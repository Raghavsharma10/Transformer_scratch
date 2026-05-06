def saveAs(self, event=None):
        """ Save the parameter settings to a user-specified file.  Any
        changes here must be coordinated with the corresponding tpar save_as
        function. """

        self.debug('Clicked Save as...')
        # On Linux Pers..Dlg causes the cwd to change, so get a copy of current
        curdir = os.getcwd()

        # The user wishes to save to a different name
        writeProtChoice = self._writeProtectOnSaveAs
        if capable.OF_TKFD_IN_EPAR:
            # Prompt using native looking dialog
            fname = asksaveasfilename(parent=self.top,
                    title='Save Parameter File As',
                    defaultextension=self._defSaveAsExt,
                    initialdir=os.path.dirname(self._getSaveAsFilter()))
        else:
            # Prompt. (could use tkinter's FileDialog, but this one is prettier)
            # initWProtState is only used in the 1st call of a session
            from . import filedlg
            fd = filedlg.PersistSaveFileDialog(self.top,
                         "Save Parameter File As", self._getSaveAsFilter(),
                         initWProtState=writeProtChoice)
            if fd.Show() != 1:
                fd.DialogCleanup()
                os.chdir(curdir) # in case file dlg moved us
                return
            fname = fd.GetFileName()
            writeProtChoice = fd.GetWriteProtectChoice()
            fd.DialogCleanup()

        if not fname: return # canceled

        # First check the child parameters, aborting save if
        # invalid entries were encountered
        if self.checkSetSaveChildren():
            os.chdir(curdir) # in case file dlg moved us
            return

        # Run any subclass-specific steps right before the save
        self._saveAsPreSave_Hook(fname)

        # Verify all the entries (without save), keeping track of the invalid
        # entries which have been reset to their original input values
        self.badEntriesList = self.checkSetSaveEntries(doSave=False)

        # If there were invalid entries, prepare the message dialog
        if self.badEntriesList:
            ansOKCANCEL = self.processBadEntries(self.badEntriesList,
                          self.taskName)
            if not ansOKCANCEL:
                os.chdir(curdir) # in case file dlg moved us
                return

        # If there were no invalid entries or the user says OK, finally
        # save to their stated file.  Since we have already processed the
        # bad entries, there should be none returned.
        mstr = "TASKMETA: task="+self.taskName+" package="+self.pkgName
        if self.checkSetSaveEntries(doSave=True, filename=fname, comment=mstr,
                                    set_ro=writeProtChoice,
                                    overwriteRO=True):
            os.chdir(curdir) # in case file dlg moved us
            raise Exception("Unexpected bad entries for: "+self.taskName)

        # Run any subclass-specific steps right after the save
        self._saveAsPostSave_Hook(fname)

        os.chdir(curdir)