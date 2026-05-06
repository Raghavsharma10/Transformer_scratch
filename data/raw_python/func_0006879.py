def match(self, s):
      """ Matching the pattern to the input string, returns True/False and
          saves the matched string in the internal list
      """
      if self.re.match(s):
         self.list.append(s)
         return True
      else: return False