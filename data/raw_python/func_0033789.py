def write_sub_files(self):
    """
    Write all the submit files used by the dag to disk. Each submit file is
    written to the file name set in the CondorJob.
    """
    if not self.__nodes_finalized:
      for node in self.__nodes:
        node.finalize()
    if not self.is_dax():
      for job in self.__jobs:
        job.write_sub_file()