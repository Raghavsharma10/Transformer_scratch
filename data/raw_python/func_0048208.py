def start_range(self):
      """Similar to the junction range but don't need to check for leftmost or rightmost"""
      if len(self._exons) == 0: return None
      return GenomicRange(self._exons[0].chr,
             min([x.start for x in self._exons]),# must be part of junction
             max([x.start for x in self._exons]))