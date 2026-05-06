def set_start(self,time,pad = None):
    """
    Set the start time of the datafind query.
    @param time: GPS start time of query.
    """
    if pad:
      self.add_var_opt('gps-start-time', int(time)-int(pad))
    else:
      self.add_var_opt('gps-start-time', int(time))
    self.__start = time
    self.__set_output()