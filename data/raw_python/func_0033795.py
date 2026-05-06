def set_trig_end(self,time,pass_to_command_line=True):
    """
    Set the trig end time of the analysis node by setting a --trig-end-time
    option to the node when it is executed.
    @param time: trig end time of job.
    @bool pass_to_command_line: add trig-end-time as a variable option.
    """
    if pass_to_command_line:
      self.add_var_opt('trig-end-time',time)
    self.__trig_end = time