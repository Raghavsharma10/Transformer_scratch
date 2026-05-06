def alignUnalignedSeqsFromFile(filename,WorkingDir=None,SuppressStderr=None,\
    SuppressStdout=None):
    """Aligns unaligned sequences from some file (file should be right format)

    filename: string, the filename of the file containing the sequences
        to be aligned in a valid format.
    """
    app = Clustalw(WorkingDir=WorkingDir,SuppressStderr=SuppressStderr,\
        SuppressStdout=SuppressStdout)
    return app(filename)