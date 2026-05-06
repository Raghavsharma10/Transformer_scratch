def coalesce(self):
    """
    Coalesces any adjacent ScienceSegments. Returns the number of
    ScienceSegments in the coalesced list.
    """

    # check for an empty list
    if len(self) == 0:
      return 0

    # sort the list of science segments
    self.__sci_segs.sort()

    # coalesce the list, checking each segment for validity as we go
    outlist = []
    ostop = -1

    for seg in self:
      start = seg.start()
      stop = seg.end()
      id = seg.id()
      if start > ostop:
        # disconnected, so flush out the existing segment (if any)
        if ostop >= 0:
          x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
          outlist.append(x)
        ostart = start
        ostop = stop
      elif stop > ostop:
        # extend the current segment
        ostop = stop

    # flush out the final segment (if any)
    if ostop >= 0:
      x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
      outlist.append(x)

    self.__sci_segs = outlist
    return len(self)