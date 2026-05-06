def get_SAM(self,min_intron_size=68):
    """Get a SAM object representation of the alignment.

    :returns: SAM representation
    :rtype: SAM

    """
    from seqtools.format.sam import SAM
    #ar is target then query
    qname = self.alignment_ranges[0][1].chr
    flag = 0
    if self.strand == '-': flag = 16
    rname = self.alignment_ranges[0][0].chr
    pos = self.alignment_ranges[0][0].start
    mapq = 255
    cigar = self.construct_cigar(min_intron_size)
    rnext = '*'
    pnext = 0
    tlen = 0 # possible to set if we have a reference
    if self._options.reference:
       if rname in self._options.reference: 
          tlen = len(self._options.reference[rname])
    seq = self.query_sequence
    if not seq: seq = '*'
    qual = self.query_quality
    if not qual: qual = '*'
    #seq = '*'
    #qual = '*'
    if self.strand == '-':
      seq = rc(seq)
      qual = qual[::-1]
    ln = qname + "\t" + str(flag) + "\t" + rname + "\t" + \
         str(pos) + "\t" + str(mapq) + "\t" + cigar + "\t" + \
         rnext + "\t" + str(pnext) + "\t" + str(tlen) + "\t" + \
         seq + "\t" + qual
    return SAM(ln,reference=self._reference)