def union(self, other):
    """
    Replaces the ScienceSegments contained in this instance of ScienceData
    with the union of those in the instance other. Returns the number of
    ScienceSegments in the union.
    @param other: ScienceData to use to generate the intersection
    """

    # we only deal with the case of two lists here
    length1 = len(self)
    length2 = len(other)

    # initialize list of output segments
    ostart = -1
    seglist = []

    i1 = -1
    i2 = -1
    start1 = -1
    start2 = -1
    id = -1

    while 1:
      # if necessary, get a segment from list 1
      if start1 == -1:
        i1 += 1
        if i1 < length1:
          start1 = self[i1].start()
          stop1 = self[i1].end()
          id = self[i1].id()
        elif i2 == length2:
          break

      # if necessary, get a segment from list 2
      if start2 == -1:
        i2 += 1
        if i2 < length2:
          start2 = other[i2].start()
          stop2 = other[i2].end()
        elif i1 == length1:
          break

      # pick the earlier segment from the two lists
      if start1 > -1 and ( start2 == -1 or start1 <= start2):
        ustart = start1
        ustop = stop1
        # mark this segment has having been consumed
        start1 = -1
      elif start2 > -1:
        ustart = start2
        ustop = stop2
        # mark this segment has having been consumed
        start2 = -1
      else:
        break

      # if the output segment is blank, initialize it; otherwise, see
      # whether the new segment extends it or is disjoint
      if ostart == -1:
        ostart = ustart
        ostop = ustop
      elif ustart <= ostop:
        if ustop > ostop:
          # this extends the output segment
          ostop = ustop
        else:
          # This lies entirely within the current output segment
          pass
      else:
         # flush the current output segment, and replace it with the
         # new segment
         x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
         seglist.append(x)
         ostart = ustart
         ostop = ustop

    # flush out the final output segment (if any)
    if ostart != -1:
      x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
      seglist.append(x)

    self.__sci_segs = seglist
    return len(self)