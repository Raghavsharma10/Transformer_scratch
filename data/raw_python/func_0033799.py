def set_ifo_tag(self,ifo_tag,pass_to_command_line=True):
    """
    Set the ifo tag that is passed to the analysis code.
    @param ifo_tag: a string to identify one or more IFOs
    @bool pass_to_command_line: add ifo-tag as a variable option.
    """
    self.__ifo_tag = ifo_tag
    if pass_to_command_line:
      self.add_var_opt('ifo-tag', ifo_tag)