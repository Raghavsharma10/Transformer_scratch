def _set_seq(self,sequence,start=0):
    """Set the sequence from a start position for the length of the sequence"""
    if start+len(sequence) > self._slen: 
      sys.stderr.write("Error not long enough to add\n")
      sys.exit()
    z = 0
    for i in xrange(start, start+len(sequence)):
      self._set_nt(sequence[z],i)
      z+=1