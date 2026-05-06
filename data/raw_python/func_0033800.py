def set_user_tag(self,usertag,pass_to_command_line=True):
    """
    Set the user tag that is passed to the analysis code.
    @param user_tag: the user tag to identify the job
    @bool pass_to_command_line: add user-tag as a variable option.
    """
    self.__user_tag = usertag
    if pass_to_command_line:
      self.add_var_opt('user-tag', usertag)