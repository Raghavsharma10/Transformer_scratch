def dirBrowser(self):
        """Invoke a tkinter directory dialog"""
        if capable.OF_TKFD_IN_EPAR:
            fname = askdirectory(parent=self.entry, title="Select Directory")
        else:
            raise NotImplementedError('Fix popupChoices() logic.')

        if not fname:
            return # canceled

        self.choice.set(fname)
        # don't select when we go back to widget to reduce risk of
        # accidentally typing over the filename
        self.lastSelection = None