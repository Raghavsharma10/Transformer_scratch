def set_start(self,t):
    """
    Override the GPS start time (and set the duration) of this ScienceSegment.
    @param t: new GPS start time.
    """
    self.__dur += self.__start - t
    self.__start = t