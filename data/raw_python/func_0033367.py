def insert_sequences_into_tree(aln, moltype, params={},
                                           write_log=True):
    """Returns a tree from Alignment object aln.

    aln: an xxx.Alignment object, or data that can be used to build one.

    moltype: cogent.core.moltype.MolType object

    params: dict of parameters to pass in to the RAxML app controller.

    The result will be an xxx.Alignment object, or None if tree fails.
    """

    # convert aln to phy since seq_names need fixed to run through pplacer

    new_aln=get_align_for_phylip(StringIO(aln))

    # convert aln to fasta in case it is not already a fasta file
    aln2 = Alignment(new_aln)
    seqs = aln2.toFasta()

    ih = '_input_as_multiline_string'

    pplacer_app = Pplacer(params=params,
                      InputHandler=ih,
                      WorkingDir=None,
                      SuppressStderr=False,
                      SuppressStdout=False)

    pplacer_result = pplacer_app(seqs)

    # write a log file
    if write_log:
        log_fp = join(params["--out-dir"],'log_pplacer_' + \
                      split(get_tmp_filename())[-1])
        log_file=open(log_fp,'w')
        log_file.write(pplacer_result['StdOut'].read())
        log_file.close()

    # use guppy to convert json file into a placement tree
    guppy_params={'tog':None}

    new_tree=build_tree_from_json_using_params(pplacer_result['json'].name, \
                                               output_dir=params['--out-dir'], \
                                               params=guppy_params)

    pplacer_result.cleanUp()

    return new_tree