def get_PSL(self,min_intron_size=68):
    """Get a PSL object representation of the alignment.

    :returns: PSL representation
    :rtype: PSL

    """
    from seqtools.format.psl import PSL
    matches = sum([x[0].length for x in self.alignment_ranges]) # 1. Matches - Number of matching bases that aren't repeats
    stats = AlignmentStats(matches,0,0,0,0,0,0,0)
    sub = self.query_sequence
    ref = self._options.reference
    if ref and sub:
      stats = self._analyze_alignment(min_intron_size=min_intron_size)
    strand = self.strand # 9. strand 
    qName = self.alignment_ranges[0][1].chr # 10. qName - Query sequence name
    qSize = self.query_sequence_length
    qStart = self.alignment_ranges[0][1].start-1
    qEnd = self.alignment_ranges[-1][1].end
    tName = self.alignment_ranges[0][0].chr
    tSize = self.target_sequence_length
    tStart = self.alignment_ranges[0][0].start-1
    tEnd = self.alignment_ranges[-1][0].end
    blockCount = len(self.alignment_ranges)
    blockSizes = ','.join([str(x[0].length) for x in self.alignment_ranges])+','
    qStarts = ','.join([str(x[1].start-1) for x in self.alignment_ranges])+','
    tStarts = ','.join([str(x[0].start-1) for x in self.alignment_ranges])+','

    psl_string = str(stats.matches)+"\t"+\
    str(stats.misMatches)+"\t"+\
    str(stats.repMatches)+"\t"+\
    str(stats.nCount)+"\t"+\
    str(stats.qNumInsert)+"\t"+\
    str(stats.qBaseInsert)+"\t"+\
    str(stats.tNumInsert)+"\t"+\
    str(stats.tBaseInsert)+"\t"+\
    strand+"\t"+\
    qName+"\t"+\
    str(qSize)+"\t"+\
    str(qStart)+"\t"+\
    str(qEnd)+"\t"+\
    tName+"\t"+\
    str(tSize)+"\t"+\
    str(tStart)+"\t"+\
    str(tEnd)+"\t"+\
    str(blockCount)+"\t"+\
    blockSizes+"\t"+\
    qStarts+"\t"+\
    tStarts
    return PSL(psl_string,PSL.Options(query_sequence=self.query_sequence,reference=self._options.reference,query_quality=self.query_quality))