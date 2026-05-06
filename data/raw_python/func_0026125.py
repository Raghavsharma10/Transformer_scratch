def fileBrowser(self):
        """Invoke a tkinter file dialog"""
        if capable.OF_TKFD_IN_EPAR:
           fname = askopenfilename(parent=self.entry, title="Select File")
        else:
            from . import filedlg
            self.fd = filedlg.PersistLoadFileDialog(self.entry,
                              "Select File", "*")
            if self.fd.Show() != 1:
                self.fd.DialogCleanup()
                return
            fname = self.fd.GetFileName()
            self.fd.DialogCleanup()
        if not fname: return # canceled

        self.choice.set(fname)
        # don't select when we go back to widget to reduce risk of
        # accidentally typing over the filename
        self.lastSelection = None