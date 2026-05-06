def target_range(self):
    """Get the range on the target strand

    :return: target range
    :rtype: GenomicRange
    """
    if not self.is_aligned(): return None
    if self._target_range: return self._target_range # check cache
    global _sam_cigar_target_add
    tlen = sum([x[0] for x in self.cigar_array if _sam_cigar_target_add.match(x[1])])
    self._target_range = GenomicRange(self.entries.rname,self.entries.pos,self.entries.pos+tlen-1)
    return self._target_range