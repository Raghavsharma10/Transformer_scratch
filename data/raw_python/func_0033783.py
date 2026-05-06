def write_input_files(self, fh):
    """
    Write as a comment into the DAG file the list of input files
    for this DAG node.

    @param fh: descriptor of open DAG file.
    """
    for f in self.__input_files:
        print >>fh, "## Job %s requires input file %s" % (self.__name, f)