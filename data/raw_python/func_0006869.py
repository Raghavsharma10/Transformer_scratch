def print_out(self, *lst):
      """ Print list of strings to the predefined stdout. """
      self.print2file(self.stdout, True, True, *lst)