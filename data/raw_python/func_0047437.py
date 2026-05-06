def actual_query_range(self):
    """This is the actual query range for the positive strand

    :returns: Range of query positive strand covered
    :rtype: GenomicRange

    """
    a = self.alignment_ranges
    #return GenomicRange(a[0][1].chr,a[0][1].start,a[-1][1].end,self.get_strand())
    if self.get_strand() == '+':
      return GenomicRange(a[0][1].chr,a[0][1].start,a[-1][1].end,self.get_strand())
    #must be - strand
    return GenomicRange(a[0][1].chr,self.query_sequence_length-a[-1][1].end+1,self.query_sequence_length-a[0][1].start+1,dir=self.strand)