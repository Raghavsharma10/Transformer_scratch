def cmp(self,range2,overlap_size=0):
    """the comparitor for ranges

     * return 1 if greater than range2
     * return -1 if less than range2
     * return 0 if overlapped

    :param range2:
    :param overlap_size: allow some padding for an 'equal' comparison (default 0)
    :type range2: GenomicRange
    :type overlap_size: int

    """
    if self.overlaps(range2,padding=overlap_size): return 0
    if self.chr < range2.chr: return -1
    elif self.chr > range2.chr: return 1
    elif self.end < range2.start: return -1
    elif self.start > range2.end: return 1
    sys.stderr.write("ERROR: cmp function unexpcted state\n")
    sys.exit()
    return 0