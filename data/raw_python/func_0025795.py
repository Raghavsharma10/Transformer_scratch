def freshenFocus(self):
        """ Did something which requires a new look.  Move scrollbar up.
            This often needs to be delayed a bit however, to let other
            events in the queue through first. """
        self.top.update_idletasks()
        self.top.after(10, self.setViewAtTop)