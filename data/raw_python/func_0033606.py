def alignUnalignedSeqs(seqs,add_seq_names=True,WorkingDir=None,\
    SuppressStderr=None,SuppressStdout=None):
    """Aligns unaligned sequences

    seqs: either list of sequence objects or list of strings
    add_seq_names: boolean. if True, sequence names are inserted in the list
        of sequences. if False, it assumes seqs is a list of lines of some
        proper format that the program can handle
    """
    if add_seq_names:
        app = Clustalw(InputHandler='_input_as_seqs',\
            WorkingDir=WorkingDir,SuppressStderr=SuppressStderr,\
            SuppressStdout=SuppressStdout)
    else:
        app = Clustalw(InputHandler='_input_as_lines',\
            WorkingDir=WorkingDir,SuppressStderr=SuppressStderr,\
            SuppressStdout=SuppressStdout)
    return app(seqs)