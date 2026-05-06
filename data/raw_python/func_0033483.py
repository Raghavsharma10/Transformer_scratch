def aln_tree_seqs(seqs,
                 input_handler=None,
                 tree_type='neighborjoining',
                 params={},
                 add_seq_names=True,
                 WorkingDir=tempfile.gettempdir(),
                 SuppressStderr=None,
                 SuppressStdout=None,
                 max_hours=5.0,
                 constructor=PhyloNode,
                 clean_up=True
                 ):
    """Muscle align sequences and report tree from iteration2.

    Unlike cluster_seqs, returns tree2 which is the tree made during the
    second muscle iteration (it should be more accurate that the cluster from
    the first iteration which is made fast based on  k-mer words)

    seqs: either file name or list of sequence objects or list of strings or
        single multiline string containing sequences.
    tree_type: can be either neighborjoining (default) or upgmb for UPGMA
    clean_up: When true, will clean up output files
    """

    params["-maxhours"] = max_hours
    if tree_type:
        params["-cluster2"] = tree_type
    params["-tree2"] = get_tmp_filename(WorkingDir)
    params["-out"] = get_tmp_filename(WorkingDir)

    muscle_res = muscle_seqs(seqs,
                 input_handler=input_handler,
                 params=params,
                 add_seq_names=add_seq_names,
                 WorkingDir=WorkingDir,
                 SuppressStderr=SuppressStderr,
                 SuppressStdout=SuppressStdout)
    tree = DndParser(muscle_res["Tree2Out"], constructor=constructor)
    aln = [line for line in muscle_res["MuscleOut"]]

    if clean_up:
        muscle_res.cleanUp()
    return tree, aln