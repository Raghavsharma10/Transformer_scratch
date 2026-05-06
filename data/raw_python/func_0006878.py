def match(self, s):
      """ Matches the string to the stored regular expression, and stores all
      groups in mathches. Returns False on negative match. """
      self.matches = self.re.search(s)
      return self.matches