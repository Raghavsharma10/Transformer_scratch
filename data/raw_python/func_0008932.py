def find_neighbors(neighbors, coords, I, source_files, f, sides):
    """Find the tile neighbors based on filenames

    Parameters
    -----------
    neighbors : dict
        Dictionary that stores the neighbors. Format is
        neighbors["source_file_name"]["side"] = "neighbor_source_file_name"
    coords : list
        List of coordinates determined from the filename.
        See :py:func:`utils.parse_fn`
    I : array
        Sort index. Different sorting schemes will speed up when neighbors
        are found
    source_files : list
        List of strings of source file names
    f : callable
        Function that determines if two tiles are neighbors based on their
        coordinates. f(c1, c2) returns True if tiles are neighbors
    sides : list
        List of 2 strings that give the "side" where tiles are neighbors.

    Returns
    -------
    neighbors : dict
        Dictionary of neighbors

    Notes
    -------
    For example, if Tile1 is to the left of Tile2, then
    neighbors['Tile1']['right'] = 'Tile2'
    neighbors['Tile2']['left'] = 'Tile1'
    """
    for i, c1 in enumerate(coords):
        me = source_files[I[i]]
        # If the left neighbor has already been found...
        if neighbors[me][sides[0]] != '':
            continue
        # could try coords[i:] (+ fixes) for speed if it becomes a problem
        for j, c2 in enumerate(coords):
            if f(c1, c2):
                # then tiles are neighbors neighbors
                neigh = source_files[I[j]]
                neighbors[me][sides[0]] = neigh
                neighbors[neigh][sides[1]] = me
                break
    return neighbors