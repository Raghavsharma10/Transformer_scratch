def build_tree_from_alignment(aln, moltype=DNA, best_tree=False, params=None):
    """Returns a tree from Alignment object aln.

    aln: a cogent.core.alignment.Alignment object, or data that can be used
    to build one.

    moltype: cogent.core.moltype.MolType object

    best_tree: unsupported

    params: dict of parameters to pass in to the Muscle app controller.

    The result will be an cogent.core.tree.PhyloNode object, or None if tree
    fails.
    """
    # Create instance of app controller, enable tree, disable alignment
    app = Muscle(InputHandler='_input_as_multiline_string', params=params, \
                   WorkingDir=tempfile.gettempdir())

    app.Parameters['-clusteronly'].on()
    app.Parameters['-tree1'].on(get_tmp_filename(app.WorkingDir))
    app.Parameters['-seqtype'].on(moltype.label)

    seq_collection = SequenceCollection(aln, MolType=moltype)

    #Create mapping between abbreviated IDs and full IDs
    int_map, int_keys = seq_collection.getIntMap()
    #Create SequenceCollection from int_map.
    int_map = SequenceCollection(int_map,MolType=moltype)


    # Collect result
    result = app(int_map.toFasta())

    # Build tree
    tree = DndParser(result['Tree1Out'].read(), constructor=PhyloNode)

    for tip in tree.tips():
        tip.Name = int_keys[tip.Name]

    # Clean up
    result.cleanUp()
    del(seq_collection, app, result)

    return tree