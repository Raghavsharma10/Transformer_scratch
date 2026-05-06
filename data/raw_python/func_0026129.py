def setActiveState(self, active):
        """ Use this to enable or disable (grey out) a parameter. """
        st = DISABLED
        if active: st = NORMAL
        self.entry.configure(state=st)
        self.inputLabel.configure(state=st)
        self.promptLabel.configure(state=st)