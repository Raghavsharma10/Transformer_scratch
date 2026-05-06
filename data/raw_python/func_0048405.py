def get_range_start_line_number(self,rng):
    """
    .. warning:: not implemented
    """
    sys.stderr.write("error unimplemented get_range_start_line\n")
    sys.exit()
    for i in range(0,len(self._lines)):
      if rng.cmp(self._lines[i]['rng'])==0: return i+1
    return None