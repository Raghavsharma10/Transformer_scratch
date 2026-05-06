def add_file_arg(self, filename):
    """
    Add a file argument to the executable. Arguments are appended after any
    options and their order is guaranteed. Also adds the file name to the
    list of required input data for this job.
    @param filename: file to add as argument.
    """
    self.__arguments.append(filename)
    if filename not in self.__input_files:
      self.__input_files.append(filename)