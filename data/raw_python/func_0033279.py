def blastp(seqs, blast_db="nr", e_value="1e-20", max_hits=200,
           working_dir=tempfile.gettempdir(), blast_mat_root=None,
           extra_params={}):
    """
    Returns BlastResult from input seqs, using blastp.

    Need to add doc string
    """

    # set up params to use with blastp
    params = {
        # matrix
        "-M":"BLOSUM62",

        # max procs
        "-a":"1",

        # expectation
        "-e":e_value,

        # max seqs to show
        "-b":max_hits,

        # max one line descriptions
        "-v":max_hits,

        # program
        "-p":"blastp"
    }
    params.update(extra_params)

    # blast
    blast_res =  blast_seqs(seqs,
        Blastall,
        blast_mat_root=blast_mat_root,
        blast_db=blast_db,
        params=params,
        add_seq_names=False,
        WorkingDir=working_dir
        )

    # get prot id map
    if blast_res['StdOut']:
        lines = [x for x in blast_res['StdOut']]
        return BlastResult(lines)

    return None