def make_short_chunks_from_unused(
    self,min_length,overlap=0,play=0,sl=0,excl_play=0):
    """
    Create a chunk that uses up the unused data in the science segment
    @param min_length: the unused data must be greater than min_length to make a
    chunk.
    @param overlap: overlap between chunks in seconds.
    @param play: if true, only generate chunks that overlap with S2 playground data.
    @param sl: slide by sl seconds before determining playground data.
    @param excl_play: exclude the first excl_play second from the start and end
    of the chunk when computing if the chunk overlaps with playground.
    """
    for seg in self.__sci_segs:
      if seg.unused() > min_length:
        start = seg.end() - seg.unused() - overlap
        end = seg.end()
        length = start - end
        if (not play) or (play and (((end-sl-excl_play-729273613)%6370) <
        (600+length-2*excl_play))):
          seg.add_chunk(start, end, start)
        seg.set_unused(0)