def alignTwoAlignments(aln1,aln2,outfile,WorkingDir=None,SuppressStderr=None,\
    SuppressStdout=None):
    """Aligns two alignments. Individual sequences are not realigned

    aln1: string, name of file containing the first alignment
    aln2: string, name of file containing the second alignment
    outfile: you're forced to specify an outfile name, because if you don't
        aln1 will be overwritten. So, if you want aln1 to be overwritten, you
        should specify the same filename.
    WARNING: a .dnd file is created with the same prefix as aln1. So an
    existing dendrogram might get overwritten.
    """
    app = Clustalw({'-profile':None,'-profile1':aln1,\
        '-profile2':aln2,'-outfile':outfile},SuppressStderr=\
        SuppressStderr,WorkingDir=WorkingDir,SuppressStdout=SuppressStdout)
    app.Parameters['-align'].off()
    return app()