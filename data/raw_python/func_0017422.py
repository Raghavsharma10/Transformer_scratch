def _read_newick_from_string(nw, root_node, matcher, formatcode):
    """ Reads a newick string in the New Hampshire format. """
    if nw.count('(') != nw.count(')'):
        raise NewickError('Parentheses do not match. Broken tree structure?')

    # white spaces and separators are removed
    nw = re.sub("[\n\r\t]+", "", nw)

    current_parent = None
    # Each chunk represents the content of a parent node, and it could contain
    # leaves and closing parentheses.
    # We may find:
    # leaf, ..., leaf,
    # leaf, ..., leaf))),
    # leaf)), leaf, leaf))
    # leaf))
    # ) only if formatcode == 100

    for chunk in nw.split("(")[1:]:
        # If no node has been created so far, this is the root, so use the node.
        current_parent = root_node if current_parent is None else current_parent.add_child()

        subchunks = [ch.strip() for ch in chunk.split(",")]
        # We should expect that the chunk finished with a comma (if next chunk
        # is an internal sister node) or a subchunk containing closing parenthesis until the end of the tree.
        #[leaf, leaf, '']
        #[leaf, leaf, ')))', leaf, leaf, '']
        #[leaf, leaf, ')))', leaf, leaf, '']
        #[leaf, leaf, ')))', leaf), leaf, 'leaf);']
        if subchunks[-1] != '' and not subchunks[-1].endswith(';'):
            raise NewickError('Broken newick structure at: %s' %chunk)

        # lets process the subchunks. Every closing parenthesis will close a
        # node and go up one level.
        for i, leaf in enumerate(subchunks):
            if leaf.strip() == '' and i == len(subchunks) - 1:
                continue # "blah blah ,( blah blah"
            closing_nodes = leaf.split(")")

            # first part after splitting by ) always contain leaf info
            _read_node_data(closing_nodes[0], current_parent, "leaf", matcher, formatcode)

            # next contain closing nodes and data about the internal nodes.
            if len(closing_nodes)>1:
                for closing_internal in closing_nodes[1:]:
                    closing_internal =  closing_internal.rstrip(";")
                    # read internal node data and go up one level
                    _read_node_data(closing_internal, current_parent, "internal", matcher, formatcode)
                    current_parent = current_parent.up
    return root_node