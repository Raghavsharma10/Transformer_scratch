def addSeqsToAlignment(aln1,seqs,outfile,WorkingDir=None,SuppressStderr=None,\
        SuppressStdout=None):
    """Aligns sequences from second profile against first profile

    aln1: string, name of file containing the alignment
    seqs: string, name of file containing the sequences that should be added
        to the alignment.
    opoutfile: string, name of the output file (the new alignment)
    """
    app = Clustalw({'-sequences':None,'-profile1':aln1,\
        '-profile2':seqs,'-outfile':outfile},SuppressStderr=\
        SuppressStderr,WorkingDir=WorkingDir, SuppressStdout=SuppressStdout)

    app.Parameters['-align'].off()
    return app()