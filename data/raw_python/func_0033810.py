def intersection(self, other):
    """
    Replaces the ScienceSegments contained in this instance of ScienceData
    with the intersection of those in the instance other. Returns the number
    of segments in the intersection.
    @param other: ScienceData to use to generate the intersection
    """

    # we only deal with the case of two lists here
    length1 = len(self)
    length2 = len(other)

    # initialize list of output segments
    ostart = -1
    outlist = []
    iseg2 = -1
    start2 = -1
    stop2 = -1

    for seg1 in self:
      start1 = seg1.start()
      stop1 = seg1.end()
      id = seg1.id()

      # loop over segments from the second list which overlap this segment
      while start2 < stop1:
        if stop2 > start1:
          # these overlap

          # find the overlapping range
          if start1 < start2:
            ostart = start2
          else:
            ostart = start1
          if stop1 > stop2:
            ostop = stop2
          else:
            ostop = stop1

          x = ScienceSegment(tuple([id, ostart, ostop, ostop-ostart]))
          outlist.append(x)

          if stop2 > stop1:
            break

        # step forward
        iseg2 += 1
        if iseg2 < len(other):
          seg2 = other[iseg2]
          start2 = seg2.start()
          stop2 = seg2.end()
        else:
          # pseudo-segment in the far future
          start2 = 2000000000
          stop2 = 2000000000

    # save the intersection and return the length
    self.__sci_segs = outlist
    return len(self)