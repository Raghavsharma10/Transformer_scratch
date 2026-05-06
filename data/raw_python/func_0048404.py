def get_range_start_coord(self,rng):
    """
    .. warning:: not implemented
    """
    sys.stderr.write("error unimplemented get_range_start_coord\n")
    sys.exit()
    if rng.chr not in self._chrs: return None
    for l in [self._lines[x-1] for x in self._chrs[rng.chr]]:
      ####
      y = l['rng']
      c = y.cmp(rng)
      if c > 0: return None
      if c == 0:
        x = y.get_payload()
        return [x[1],x[2]] # don't need the name
    return None