def rotate(self, log):
        """Move the current log to a new file with timestamp and create a new empty log file."""
        self.write(log, rotate=True)
        self.write({})