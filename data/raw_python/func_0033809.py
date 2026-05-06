def make_optimised_chunks(self, min_length, max_length, pad_data=0):
    """
    Splits ScienceSegments up into chunks, of a given maximum length.
    The length of the last two chunks are chosen so that the data
    utilisation is optimised.
    @param min_length: minimum chunk length.
    @param max_length: maximum chunk length.
    @param pad_data: exclude the first and last pad_data seconds of the
    segment when generating chunks
    """
    for seg in self.__sci_segs:
      # pad data if requested
      seg_start = seg.start() + pad_data
      seg_end = seg.end() - pad_data

      if seg.unused() > max_length:
        # get number of max_length chunks
        N = (seg_end - seg_start)/max_length

        # split into chunks of max_length
        for i in range(N-1):
          start = seg_start + (i * max_length)
          stop = start + max_length
          seg.add_chunk(start, stop)

        # optimise data usage for last 2 chunks
        start = seg_start + ((N-1) * max_length)
        middle = (start + seg_end)/2
        seg.add_chunk(start, middle)
        seg.add_chunk(middle, seg_end)
        seg.set_unused(0)
      elif seg.unused() > min_length:
        # utilise as single chunk
        seg.add_chunk(seg_start, seg_end)
      else:
        # no chunk of usable length
        seg.set_unused(0)