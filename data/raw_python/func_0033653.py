def build_tree_from_alignment(aln, moltype=DNA, best_tree=False, params={},\
    working_dir='/tmp'):
    """Returns a tree from Alignment object aln.

    aln: an cogent.core.alignment.Alignment object, or data that can be used
    to build one.
        -  Clearcut only accepts aligned sequences.  Alignment object used to
        handle unaligned sequences.

    moltype: a cogent.core.moltype object.
        - NOTE: If moltype = RNA, we must convert to DNA since Clearcut v1.0.8
        gives incorrect results if RNA is passed in.  'U' is treated as an
        incorrect character and is excluded from distance calculations.

    best_tree: if True (default:False), uses a slower but more accurate
    algorithm to build the tree.

    params: dict of parameters to pass in to the Clearcut app controller.

    The result will be an cogent.core.tree.PhyloNode object, or None if tree
    fails.
    """
    params['--out'] = get_tmp_filename(working_dir)

    # Create instance of app controller, enable tree, disable alignment
    app = Clearcut(InputHandler='_input_as_multiline_string', params=params, \
                   WorkingDir=working_dir, SuppressStdout=True,\
                   SuppressStderr=True)
    #Input is an alignment
    app.Parameters['-a'].on()
    #Turn off input as distance matrix
    app.Parameters['-d'].off()

    #If moltype = RNA, we must convert to DNA.
    if moltype == RNA:
        moltype = DNA

    if best_tree:
        app.Parameters['-N'].on()

    #Turn on correct moltype
    moltype_string = moltype.label.upper()
    app.Parameters[MOLTYPE_MAP[moltype_string]].on()

    # Setup mapping. Clearcut clips identifiers. We will need to remap them.
    # Clearcut only accepts aligned sequences.  Let Alignment object handle
    # unaligned sequences.
    seq_aln = Alignment(aln,MolType=moltype)
    #get int mapping
    int_map, int_keys = seq_aln.getIntMap()
    #create new Alignment object with int_map
    int_map = Alignment(int_map)

    # Collect result
    result = app(int_map.toFasta())

    # Build tree
    tree = DndParser(result['Tree'].read(), constructor=PhyloNode)
    for node in tree.tips():
        node.Name = int_keys[node.Name]

    # Clean up
    result.cleanUp()
    del(seq_aln, app, result, int_map, int_keys, params)

    return tree