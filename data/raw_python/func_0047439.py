def get_alignment_strings(self,min_intron_size=68):
    """Process the alignment to get information like
       the alignment strings for each exon. These strings are used by the pretty print.

    :returns: String representation of the alignment in an easy to read format
    :rtype: string
    """
    qseq = self.query_sequence
    if not qseq:
      sys.exit("ERROR: Query sequence must be accessable to get alignment strings\n")
      sys.exit()
    ref = self._options.reference
    qual = self.query_quality
    if not qual: 
      qual = 'I'*len(qseq) # for a placeholder quality
    if self.strand == '-': 
      qseq = rc(qseq)
      qual = qual[::-1]
    tarr = []
    qarr = []
    yarr = []
    tdone = ''
    qdone = ''
    ydone = '' #query quality
    for i in range(len(self.alignment_ranges)):
      [t,q] = self.alignment_ranges[i]
      textra = ''
      qextra = ''
      yextra = ''
      if i >= 1:
        dift = t.start-self.alignment_ranges[i-1][0].end-1
        difq = q.start-self.alignment_ranges[i-1][1].end-1
        if dift < min_intron_size:
          if dift > 0:
            textra = str(ref[t.chr][t.start-dift-1:t.start-1]).upper()
            qextra = '-'*dift
            yextra = '\0'*dift
          elif difq > 0:
            textra = '-'*difq
            qextra = qseq[q.start-difq-1:q.start-1].upper()
            yextra = qual[q.start-difq-1:q.start-1]
        else:
          tarr.append(tdone)
          qarr.append(qdone)
          yarr.append(ydone)
          tdone = ''
          qdone = ''
          ydone = ''
      tdone += textra+str(ref[t.chr][t.start-1:t.end]).upper()
      qdone += qextra+qseq[q.start-1:q.end].upper()
      ydone += yextra+qual[q.start-1:q.end]
    if len(tdone) > 0: 
      tarr.append(tdone)
      qarr.append(qdone)
      yarr.append(ydone)
    if self.query_quality == '*': yarr = [x.replace('I',' ') for x in yarr]
    #query, target, query_quality
    return [qarr,tarr,yarr]