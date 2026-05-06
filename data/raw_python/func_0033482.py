def muscle_seqs(seqs,
                 add_seq_names=False,
                 out_filename=None,
                 input_handler=None,
                 params={},
                 WorkingDir=tempfile.gettempdir(),
                 SuppressStderr=None,
                 SuppressStdout=None):
    """Muscle align list of sequences.

    seqs: a list of sequences as strings or objects, you must set add_seq_names=True
    or sequences in a multiline string, as read() from a fasta file
    or sequences in a list of lines, as readlines() from a fasta file
    or a fasta seq filename.

    == for eg, testcode for guessing
        #guess_input_handler should correctly identify input
        gih = guess_input_handler
        self.assertEqual(gih('abc.txt'), '_input_as_string')
        self.assertEqual(gih('>ab\nTCAG'), '_input_as_multiline_string')
        self.assertEqual(gih(['ACC','TGA'], True), '_input_as_seqs')
        self.assertEqual(gih(['>a','ACC','>b','TGA']), '_input_as_lines')

    == docstring for blast_seqs, apply to muscle_seqs ==
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

    Addl docs coming soon
    """

    if out_filename:
        params["-out"] = out_filename
    #else:
    #    params["-out"] = get_tmp_filename(WorkingDir)

    ih = input_handler or guess_input_handler(seqs, add_seq_names)
    muscle_app = Muscle(
                   params=params,
                   InputHandler=ih,
                   WorkingDir=WorkingDir,
                   SuppressStderr=SuppressStderr,
                   SuppressStdout=SuppressStdout)
    return muscle_app(seqs)