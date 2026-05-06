def construct_cigar(self,min_intron_size=68):
    """Create a CIGAR string from the alignment

    :returns: CIGAR string
    :rtype: string

    """

    # goes target query
    ar = self.alignment_ranges
    cig = ''
    if ar[0][1].start > 1: # soft clipped
      cig += str(ar[0][1].start-1)+'S'
    for i in range(len(ar)):
      exlen = ar[i][0].length
      cig += str(exlen)+'M'
      if i < len(ar)-1:
        # we can look at distances
        dt = ar[i+1][0].start-ar[i][0].end-1
        dq = ar[i+1][1].start-ar[i][1].end-1
        if dq > 0: cig += str(dq)+'I'
        if dt >= min_intron_size:
          cig += str(dt)+'N'
        elif dt > 0: cig += str(dt)+'D'
        elif dq <= 0:
          sys.stderr.write("ERROR cant form alignment\n")
          sys.exit()

    if ar[-1][1].end < self.query_sequence_length: # soft clipped
      cig += str(self.query_sequence_length-ar[-1][1].end)+'S'
    return cig