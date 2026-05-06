def cancel(self, event=None):
        """Function called when Cancel-button clicked.

        This method returns focus to parent, and destroys the dialog.
        """

        if self.parent != None:
            self.parent.focus_set()

        self.destroy()