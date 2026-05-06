def end_range(self):
      """Similar to the junction range but don't need to check for leftmost or rightmost"""
      if len(self._exons) == 0: return None
      return GenomicRange(self._exons[0].chr,
             min([x.end for x in self._exons]),
             max([x.end for x in self._exons]))