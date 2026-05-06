def subtract(self,range2):
    """Take another range, and list of ranges after removing range2, keep options from self

    :param range2:
    :type range2: GenomicRange
    :return: List of Genomic Ranges
    :rtype: GenomicRange[]

    """
    outranges = []
    if self.chr != range2.chr:
      outranges.append(self.copy())
      return outranges
    if not self.overlaps(range2):
      outranges.append(self.copy())
      return outranges
    if range2.start <= self.start and range2.end >= self.end:
      return outranges #delete all
    if range2.start > self.start: #left side
      nrng = type(self)(self.chr,self.start+self._start_offset,range2.start-1,self.payload,self.dir)
      outranges.append(nrng)
    if range2.end < self.end: #right side
      #ugly addon to make it work for either 0 or 1 index start
      nrng = type(self)(self.chr,range2.end+1+self._start_offset,self.end,self.payload,self.dir)
      outranges.append(nrng)
    return outranges