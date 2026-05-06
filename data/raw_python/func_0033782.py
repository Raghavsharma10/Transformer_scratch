def write_post_script(self,fh):
    """
    Write the post script for the job, if there is one
    @param fh: descriptor of open DAG file.
    """
    if self.__post_script:
      fh.write( 'SCRIPT POST ' + str(self) + ' ' + self.__post_script + ' ' +
        ' '.join(self.__post_script_args) + '\n' )