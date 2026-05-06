def _read_node_data(subnw, current_node, node_type, matcher, formatcode):
    """
    Reads a leaf node from a subpart of the original newicktree 
    """

    if node_type == "leaf" or node_type == "single":
        if node_type == "leaf":
            node = current_node.add_child()
        else:
            node = current_node
    else:
        node = current_node

    subnw = subnw.strip()
    
    if not subnw and node_type == 'leaf' and formatcode != 100:
        raise NewickError('Empty leaf node found')
    elif not subnw:
        return

    container1, container2, converterFn1, converterFn2, compiled_matcher = matcher[node_type]
    data = re.match(compiled_matcher, subnw)
    if data:
        data = data.groups()
        # This prevents ignoring errors even in flexible nodes:
        if subnw and data[0] is None and data[1] is None and data[2] is None:
            raise NewickError("Unexpected newick format '%s'" %subnw)

        if data[0] is not None and data[0] != '':
            node.add_feature(container1, converterFn1(data[0].strip()))

        if data[1] is not None and data[1] != '':
            node.add_feature(container2, converterFn2(data[1][1:].strip()))

        if data[2] is not None \
                and data[2].startswith("[&&NHX"):
            _parse_extra_features(node, data[2])
    else:
        raise NewickError("Unexpected newick format '%s' " %subnw[0:50])
    return