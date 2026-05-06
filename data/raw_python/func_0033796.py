def set_input(self,filename,pass_to_command_line=True):
    """
    Add an input to the node by adding a --input option.
    @param filename: option argument to pass as input.
    @bool pass_to_command_line: add input as a variable option.
    """
    self.__input = filename
    if pass_to_command_line:
      self.add_var_opt('input', filename)
    self.add_input_file(filename)