def set_ifo(self,ifo):
    """
    Set the ifo name to analyze. If the channel name for the job is defined,
    then the name of the ifo is prepended to the channel name obtained
    from the job configuration file and passed with a --channel-name option.
    @param ifo: two letter ifo code (e.g. L1, H1 or H2).
    """
    self.__ifo = ifo
    if self.job().channel():
      self.add_var_opt('channel-name', ifo + ':' + self.job().channel())