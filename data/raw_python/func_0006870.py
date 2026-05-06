def print_err(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stderr, False, True, *lst)