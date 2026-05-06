def write_maxjobs(self,fh,category):
    """
    Write the DAG entry for this category's maxjobs to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    @param category: tuple containing type of jobs to set a maxjobs limit for
        and the maximum number of jobs of that type to run at once.
    """
    fh.write( 'MAXJOBS ' + str(category[0]) + ' ' + str(category[1]) +  '\n' )