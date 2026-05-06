def actual_original_query_range(self):
    """ This accounts for hard clipped bases
    and a query sequence that hasnt been reverse complemented

    :return: the range covered on the original query sequence
    :rtype: GenomicRange
    """
    l = self.original_query_sequence_length
    a = self.alignment_ranges
    qname = a[0][1].chr
    qstart = a[0][1].start
    qend = a[-1][1].end
    #rng = self.get_query_range()
    start = qstart
    end = qend
    if self.strand == '-':
      end = l-(qstart-1)
      start = 1+l-(qend)
    return GenomicRange(qname,start,end,dir=self.strand)