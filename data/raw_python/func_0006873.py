def log_no_newline(self, msg):
      """ print the message to the predefined log file without newline """
      self.print2file(self.logfile, False, False, msg)