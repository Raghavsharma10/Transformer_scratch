def overlap_size(self,in_genomic_range):
    """ The size of the overlap

    :param in_genomic_range: the range to intersect
    :type in_genomic_range: GenomicRange

    :return: count of overlapping bases
    :rtype: int
    """
    if self.chr != in_genomic_range.chr:
      return 0
    if self.end < in_genomic_range.start:
      return 0
    if in_genomic_range.end < self.start:
      return 0
    if self.start > in_genomic_range.end:
      return 0
    if self.start >= in_genomic_range.start and self.end <= in_genomic_range.end:
      return self.end-self.start+1
    if self.start <= in_genomic_range.start and self.end >= in_genomic_range.end:
      return in_genomic_range.end-in_genomic_range.start+1
    if self.start <= in_genomic_range.start and self.end >= in_genomic_range.start:
      return self.end-in_genomic_range.start+1
    if self.start <= in_genomic_range.end and self.end >= in_genomic_range.end:
      return in_genomic_range.end-self.start+1
    if in_genomic_range.start <= self.start and in_genomic_range.end >= self.start:
      return in_genomic_range.end-self.start+1
    if in_genomic_range.start <= self.end and in_genomic_range.end >= self.end:
      return self.end-in_genomic_range.start+1
    sys.stderr.write("overlap_size: unprogrammed error\n")
    return 0