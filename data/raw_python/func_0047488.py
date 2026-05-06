def get_sequence(self,ref):
    """get a sequence given a reference"""
    strand = '+'
    if not self._options.direction:
      sys.stderr.write("WARNING: no strand information for the transcript\n")
    if self._options.direction: strand = self._options.direction
    seq = ''
    for e in [x.range for x in self.exons]:
      seq += str(ref[e.chr][e.start-1:e.end])
    if strand == '-':  seq = rc(seq)
    return Sequence(seq.upper())