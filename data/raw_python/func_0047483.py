def overlap_size(self,tx2):
    """Return the number of overlapping base pairs between two transcripts

    :param tx2: Another transcript
    :type tx2: Transcript
    :return: overlap size in base pairs
    :rtype: int
    """
    total = 0
    for e1 in self.exons:
      for e2 in tx2.exons:
        total += e1.overlap_size(e2)
    return total