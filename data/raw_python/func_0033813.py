def play(self):
    """
    Keep only times in ScienceSegments which are in the playground
    """

    length = len(self)

    # initialize list of output segments
    ostart = -1
    outlist = []
    begin_s2 = 729273613
    play_space = 6370
    play_len = 600

    for seg in self:
      start = seg.start()
      stop = seg.end()
      id = seg.id()

      # select first playground segment which ends after start of seg
      play_start = begin_s2+play_space*( 1 +
        int((start - begin_s2 - play_len)/play_space) )

      while play_start < stop:
        if play_start > start:
          ostart = play_start
        else:
          ostart = start


        play_stop = play_start + play_len

        if play_stop < stop:
          ostop = play_stop
        else:
          ostop = stop

        x = ScienceSegment(tuple([id, ostart, ostop, ostop-ostart]))
        outlist.append(x)

        # step forward
        play_start = play_start + play_space

    # save the playground segs and return the length
    self.__sci_segs = outlist
    return len(self)