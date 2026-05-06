def sub(self, replace, string, count=0):
      """ returns new string where the matching cases (limited by the count) in
      the string is replaced. """
      return self.re.sub(replace, string, count)