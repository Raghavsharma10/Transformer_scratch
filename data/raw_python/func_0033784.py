def write_output_files(self, fh):
    """
    Write as a comment into the DAG file the list of output files
    for this DAG node.

    @param fh: descriptor of open DAG file.
    """
    for f in self.__output_files:
        print >>fh, "## Job %s generates output file %s" % (self.__name, f)