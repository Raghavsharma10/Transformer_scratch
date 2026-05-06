def set_output(self,filename,pass_to_command_line=True):
    """
    Add an output to the node by adding a --output option.
    @param filename: option argument to pass as output.
    @bool pass_to_command_line: add output as a variable option.
    """
    self.__output = filename
    if pass_to_command_line:
      self.add_var_opt('output', filename)
    self.add_output_file(filename)