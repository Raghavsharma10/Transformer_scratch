def get_index_line(self,lnum):
    """ Take the 1-indexed line number and return its index information"""
    if lnum < 1:
      sys.stderr.write("ERROR: line number should be greater than zero\n")
      sys.exit()
    elif lnum > len(self._lines):
      sys.stderr.write("ERROR: too far this line nuber is not in index\n")
      sys.exit()
    return self._lines[lnum-1]