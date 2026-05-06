def add_file_opt(self, opt, filename):
    """
    Add a command line option to the executable. The order that the arguments
    will be appended to the command line is not guaranteed, but they will
    always be added before any command line arguments. The name of the option
    is prefixed with double hyphen and the program is expected to parse it
    with getopt_long().
    @param opt: command line option to add.
    @param value: value to pass to the option (None for no argument).
    """
    self.__options[opt] = filename
    if filename not in self.__input_files:
      self.__input_files.append(filename)