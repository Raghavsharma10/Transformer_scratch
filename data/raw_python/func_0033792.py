def set_start(self,time,pass_to_command_line=True):
    """
    Set the GPS start time of the analysis node by setting a --gps-start-time
    option to the node when it is executed.
    @param time: GPS start time of job.
    @bool pass_to_command_line: add gps-start-time as variable option.
    """
    if pass_to_command_line:
      self.add_var_opt('gps-start-time',time)
    self.__start = time
    self.__data_start = time