def blast_seqs(seqs,
                 blast_constructor,
                 blast_db=None,
                 blast_mat_root=None,
                 params={},
                 add_seq_names=True,
                 out_filename=None,
                 WorkingDir=None,
                 SuppressStderr=None,
                 SuppressStdout=None,
                 input_handler=None,
                 HALT_EXEC=False
                 ):
    """Blast list of sequences.

    seqs: either file name or list of sequence objects or list of strings or
    single multiline string containing sequences.

    WARNING: DECISION RULES FOR INPUT HANDLING HAVE CHANGED. Decision rules
    for data are as follows. If it's s list, treat as lines, unless
    add_seq_names is true (in which case treat as list of seqs). If it's a
    string, test whether it has newlines. If it doesn't have newlines, assume
    it's a filename. If it does have newlines, it can't be a filename, so
    assume it's a multiline string containing sequences.

    If you want to skip the detection and force a specific type of input
    handler, use input_handler='your_favorite_handler'.

    add_seq_names: boolean. if True, sequence names are inserted in the list
        of sequences. if False, it assumes seqs is a list of lines of some
        proper format that the program can handle
    """

    # set num keep

    if blast_db:
        params["-d"] = blast_db

    if out_filename:
        params["-o"] = out_filename

    ih = input_handler or guess_input_handler(seqs, add_seq_names)

    blast_app = blast_constructor(
                   params=params,
                   blast_mat_root=blast_mat_root,
                   InputHandler=ih,
                   WorkingDir=WorkingDir,
                   SuppressStderr=SuppressStderr,
                   SuppressStdout=SuppressStdout,
                   HALT_EXEC=HALT_EXEC)

    return blast_app(seqs)