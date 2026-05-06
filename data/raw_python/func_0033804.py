def set_end(self,t):
    """
    Override the GPS end time (and set the duration) of this ScienceSegment.
    @param t: new GPS end time.
    """
    self.__dur -= self.__end - t
    self.__end = t