def distance(self,rng):
    """The distance between two ranges.

    :param rng: another range
    :type rng: GenomicRange
    :returns: bases separting, 0 if overlapped or adjacent, -1 if on different chromsomes
    :rtype: int
    """
    if self.chr != rng.chr: return -1
    c = self.cmp(rng)
    if c == 0: return 0
    if c < 0:
      return rng.start - self.end-1
    return self.start - rng.end-1