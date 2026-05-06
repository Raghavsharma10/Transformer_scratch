def write_job(self,fh):
    """
    Write the DAG entry for this node's job to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    """
    if isinstance(self.job(),CondorDAGManJob):
      # create an external subdag from this dag
      fh.write( ' '.join(
        ['SUBDAG EXTERNAL', self.__name, self.__job.get_sub_file()]) )
      if self.job().get_dag_directory():
        fh.write( ' DIR ' + self.job().get_dag_directory() )
    else:
      # write a regular condor job
      fh.write( 'JOB ' + self.__name + ' ' + self.__job.get_sub_file() )
    fh.write( '\n')

    fh.write( 'RETRY ' + self.__name + ' ' + str(self.__retry) + '\n' )