def add_input_file(self, filename):
    """
    Add filename as a necessary input file for this DAG node.

    @param filename: input filename to add
    """
    if filename not in self.__input_files:
      self.__input_files.append(filename)