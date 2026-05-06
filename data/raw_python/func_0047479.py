def ok(self, event=None):
        """Function called when OK-button is clicked.

        This method calls check_input(), and if that returns ok it calls
        execute(), and then destroys the dialog.
        """

        if not self.check_input():
            self.initial_focus.focus_set()
            return

        self.withdraw()
        self.update_idletasks()

        try:
            self.execute()
        finally:
            self.cancel()