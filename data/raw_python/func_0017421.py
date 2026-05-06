def read_newick(newick, root_node=None, format=0):
    """ 
    Reads a newick tree from either a string or a file, and returns
    an ETE tree structure.

    A previously existent node object can be passed as the root of the
    tree, which means that all its new children will belong to the same
    class as the root (This allows to work with custom TreeNode objects).

    You can also take advantage from this behaviour to concatenate
    several tree structures.
    """

    ## check newick type as a string or filepath, Toytree parses urls to str's
    if isinstance(newick, six.string_types):   
        if os.path.exists(newick):
            if newick.endswith('.gz'):
                import gzip
                nw = gzip.open(newick).read()
            else:
                nw = open(newick, 'rU').read()
        else:
            nw = newick

        ## get re matcher for testing newick formats
        matcher = compile_matchers(formatcode=format)
        nw = nw.strip()        
        if not nw.startswith('(') and nw.endswith(';'):
            return _read_node_data(nw[:-1], root_node, "single", matcher, format)

        elif not nw.startswith('(') or not nw.endswith(';'):
            raise NewickError('Unexisting tree file or Malformed newick tree structure.')
        else:
            return _read_newick_from_string(nw, root_node, matcher, format)
    else:
        raise NewickError("'newick' argument must be either a filename or a newick string.")