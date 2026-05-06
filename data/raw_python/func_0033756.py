def add_output_file(self, filename):
    """
    Add filename as a output file for this DAG node.

    @param filename: output filename to add
    """
    if filename not in self.__output_files:
      self.__output_files.append(filename)