def _get_alignment_ranges(self):
    """A key method to extract the alignment data from the line"""
    if not self.is_aligned(): return None
    alignment_ranges = []
    cig = [x[:] for x in self.cigar_array]
    target_pos = self.entries.pos
    query_pos = 1
    while len(cig) > 0:
      c = cig.pop(0)
      if re.match('[S]$',c[1]): # hard or soft clipping
        query_pos += c[0]
      elif re.match('[ND]$',c[1]): # deleted from reference
        target_pos += c[0]
      elif re.match('[I]$',c[1]): # insertion to the reference
        query_pos += c[0]
      elif re.match('[MI=X]$',c[1]): # keep it
        t_start = target_pos
        q_start = query_pos
        target_pos += c[0]
        query_pos += c[0]
        t_end = target_pos-1
        q_end = query_pos-1
        alignment_ranges.append([GenomicRange(self.entries.rname,t_start,t_end),GenomicRange(self.entries.qname,q_start,q_end)])
    return alignment_ranges