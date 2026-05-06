def write_parents(self,fh):
    """
    Write the parent/child relations for this job to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    """
    for parent in self.__parents:
      fh.write( 'PARENT ' + str(parent) + ' CHILD ' + str(self) + '\n' )