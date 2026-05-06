def build_tree_from_distance_matrix(matrix, best_tree=False, params={},\
    working_dir='/tmp'):
    """Returns a tree from a distance matrix.

    matrix: a square Dict2D object (cogent.util.dict2d)

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
    #Turn off input as alignment
    app.Parameters['-a'].off()
    #Input is a distance matrix
    app.Parameters['-d'].on()

    if best_tree:
        app.Parameters['-N'].on()

    # Turn the dict2d object into the expected input format
    matrix_input, int_keys = _matrix_input_from_dict2d(matrix)

    # Collect result
    result = app(matrix_input)

    # Build tree
    tree = DndParser(result['Tree'].read(), constructor=PhyloNode)

    # reassign to original names
    for node in tree.tips():
        node.Name = int_keys[node.Name]

    # Clean up
    result.cleanUp()
    del(app, result, params)

    return tree