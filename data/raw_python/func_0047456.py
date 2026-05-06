def merge(self,range2): 
    """merge this bed with another bed to make a longer bed.  Returns None if on different chromosomes.

    keeps the options of this class (not range2)

    :param range2:
    :type range2: GenomicRange

    :return: bigger range with both
    :rtype: GenomicRange

    """
    if self.chr != range2.chr:
      return None
    o = type(self)(self.chr,min(self.start,range2.start)+self._start_offset,max(self.end,range2.end),self.payload,self.dir)
    return o