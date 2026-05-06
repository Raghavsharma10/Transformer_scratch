def buildTreeFromAlignment(filename,WorkingDir=None,SuppressStderr=None):
    """Builds a new tree from an existing alignment

    filename: string, name of file containing the seqs or alignment
    """
    app = Clustalw({'-tree':None,'-infile':filename},SuppressStderr=\
        SuppressStderr,WorkingDir=WorkingDir)
    app.Parameters['-align'].off()
    return app()