def bootstrap_tree_from_alignment(aln, seed=None, num_trees=None, params=None):
    """Returns a tree from Alignment object aln with bootstrap support values.

    aln: an cogent.core.alignment.Alignment object, or data that can be used
    to build one.

    seed: an interger, seed value to use

    num_trees: an integer, number of trees to bootstrap against

    params: dict of parameters to pass in to the Clustal app controller.

    The result will be an cogent.core.tree.PhyloNode object, or None if tree
    fails.

    If seed is not specifed in params, a random integer between 0-1000 is used.
    """
    # Create instance of controllor, enable bootstrap, disable alignment,tree
    app = Clustalw(InputHandler='_input_as_multiline_string', params=params, \
                   WorkingDir='/tmp')
    app.Parameters['-align'].off()
    app.Parameters['-tree'].off()

    if app.Parameters['-bootstrap'].isOff():
        if num_trees is None:
            num_trees = 1000

        app.Parameters['-bootstrap'].on(num_trees)

    if app.Parameters['-seed'].isOff():
        if seed is None:
            seed = randint(0,1000)

        app.Parameters['-seed'].on(seed)

    if app.Parameters['-bootlabels'].isOff():
        app.Parameters['-bootlabels'].on("node")

    # Setup mapping. Clustalw clips identifiers. We will need to remap them.
    seq_collection = SequenceCollection(aln)
    int_map, int_keys = seq_collection.getIntMap()
    int_map = SequenceCollection(int_map)

    # Collect result
    result = app(int_map.toFasta())

    # Build tree
    tree = DndParser(result['Tree'].read(), constructor=PhyloNode)
    for node in tree.tips():
        node.Name = int_keys[node.Name]

    # Clean up
    result.cleanUp()
    del(seq_collection, app, result, int_map, int_keys)

    return tree