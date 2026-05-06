def overlaps(self,tx2):
    """Return True if overlapping
    """
    total = 0
    for e1 in self.exons:
      for e2 in tx2.exons:
        if e1.overlap_size(e2) > 0: return True
    return False