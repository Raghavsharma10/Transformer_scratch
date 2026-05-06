def home(self):
        """Return to initial position (row=0, column=0)"""
        self.row = 0
        self.column = 0

        self.command(Command.RETURN_HOME)
        msleep(2)