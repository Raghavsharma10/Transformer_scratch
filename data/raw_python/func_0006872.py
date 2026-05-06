def log(self, *lst):
      """ Print list of strings to the predefined logfile if debug is set. and
      sets the caught_error message if an error is found
      """
      self.print2file(self.logfile, self.debug, True, *lst)
      if 'Error' in '\n'.join([str(x) for x in lst]):
         self.caught_error = '\n'.join([str(x) for x in lst])