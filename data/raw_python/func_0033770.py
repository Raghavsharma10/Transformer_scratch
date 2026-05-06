def get_output_files(self):
    """
    Return list of output files for this DAG node and its job.
    """
    output_files = list(self.__output_files)
    if isinstance(self.job(), CondorDAGJob):
      output_files = output_files + self.job().get_output_files()
    return output_files