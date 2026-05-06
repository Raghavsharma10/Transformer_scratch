def split(self, dt):
    """
      Split the segments in the list is subsegments at least as long as dt
    """
    outlist=[]
    for seg in self:
      start = seg.start()
      stop = seg.end()
      id = seg.id()

      while start < stop:
        tmpstop = start + dt
        if tmpstop > stop:
          tmpstop = stop
        elif tmpstop + dt > stop:
          tmpstop = int( (start + stop)/2 )
        x = ScienceSegment(tuple([id,start,tmpstop,tmpstop-start]))
        outlist.append(x)
        start = tmpstop

    # save the split list and return length
    self.__sci_segs = outlist
    return len(self)