def insert_sequences_into_tree(seqs, moltype, params={},
                                           write_log=True):
    """Insert sequences into Tree.

    aln: an xxx.Alignment object, or data that can be used to build one.

    moltype: cogent.core.moltype.MolType object

    params: dict of parameters to pass in to the RAxML app controller.

    The result will be a tree.
    """

    ih = '_input_as_multiline_string'

    raxml_app = Raxml(params=params,
                      InputHandler=ih,
                      WorkingDir=None,
                      SuppressStderr=False,
                      SuppressStdout=False,
                      HALT_EXEC=False)

    raxml_result = raxml_app(seqs)

    # write a log file
    if write_log:
        log_fp = join(params["-w"],'log_raxml_'+split(get_tmp_filename())[-1])
        log_file=open(log_fp,'w')
        log_file.write(raxml_result['StdOut'].read())
        log_file.close()

    '''
    # getting setup since parsimony doesn't output tree..only jplace, however
    # it is currently corrupt

    # use guppy to convert json file into a placement tree
    guppy_params={'tog':None}

    new_tree=build_tree_from_json_using_params(raxml_result['json'].name, \
                                               output_dir=params["-w"], \
                                               params=guppy_params)
    '''

    # get tree from 'Result Names'
    new_tree=raxml_result['Result'].readlines()
    filtered_tree=re.sub('\[I\d+\]','',str(new_tree))
    tree = DndParser(filtered_tree, constructor=PhyloNode)

    raxml_result.cleanUp()

    return tree