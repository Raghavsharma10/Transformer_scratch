def intersect(self,range2):
    """Return the chunk they overlap as a range.

    options is passed to result from this object

    :param range2:
    :type range2: GenomicRange

    :return: Range with the intersecting segement, or None if not overlapping
    :rtype: GenomicRange

    """
    if not self.overlaps(range2): return None
    return type(self)(self.chr,max(self.start,range2.start)+self._start_offset,min(self.end,range2.end),self.payload,self.dir)