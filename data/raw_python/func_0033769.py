def get_input_files(self):
    """
    Return list of input files for this DAG node and its job.
    """
    input_files = list(self.__input_files)
    if isinstance(self.job(), CondorDAGJob):
      input_files = input_files + self.job().get_input_files()
    return input_files