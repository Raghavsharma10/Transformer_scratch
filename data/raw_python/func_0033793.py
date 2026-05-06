def set_end(self,time,pass_to_command_line=True):
    """
    Set the GPS end time of the analysis node by setting a --gps-end-time
    option to the node when it is executed.
    @param time: GPS end time of job.
    @bool pass_to_command_line: add gps-end-time as variable option.
    """
    if pass_to_command_line:
      self.add_var_opt('gps-end-time',time)
    self.__end = time
    self.__data_end = time