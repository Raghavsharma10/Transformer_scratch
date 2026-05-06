def overlaps(self,in_genomic_range,padding=0):
    """do the ranges overlap?

    :param in_genomic_range: range to compare to
    :param padding: add to the ends this many (default 0)
    :type in_genomic_range: GenomicRange
    :type padding: int

    :return: True if they overlap
    :rtype: bool

    """
    if padding > 0:
      in_genomic_range = GenomicRange(in_genomic_range.chr,max([1,in_genomic_range.start-padding]),in_genomic_range.end+padding)
    if self.chr != in_genomic_range.chr:
      return False
    if self.end < in_genomic_range.start:
      return False
    if in_genomic_range.end < self.start:
      return False
    if self.start > in_genomic_range.end:
      return False
    if in_genomic_range.start > self.end:
      return False
    if self.start <= in_genomic_range.start and self.end >= in_genomic_range.start:
      return True
    if self.start <= in_genomic_range.end and self.end >= in_genomic_range.end:
      return True
    if self.start >= in_genomic_range.start and self.end <= in_genomic_range.end:
      return True
    if self.start <= in_genomic_range.start and self.end >= in_genomic_range.end:
      return True
    if in_genomic_range.start <= self.start and in_genomic_range.end >= self.start:
      return True
    if in_genomic_range.start <= self.end and in_genomic_range.end >= self.end:
      return True
    sys.stderr.write("overlaps: unprogrammed error\n")
    return False