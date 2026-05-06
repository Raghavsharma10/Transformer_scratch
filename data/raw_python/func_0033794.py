def set_trig_start(self,time,pass_to_command_line=True):
    """
    Set the trig start time of the analysis node by setting a
    --trig-start-time option to the node when it is executed.
    @param time: trig start time of job.
    @bool pass_to_command_line: add trig-start-time as a variable option.
    """
    if pass_to_command_line:
      self.add_var_opt('trig-start-time',time)
    self.__trig_start = time