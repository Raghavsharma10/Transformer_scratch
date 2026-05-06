def _matrix_input_from_dict2d(matrix):
    """makes input for running clearcut on a matrix from a dict2D object"""
    #clearcut truncates names to 10 char- need to rename before and
    #reassign after

    #make a dict of env_index:full name
    int_keys = dict([('env_' + str(i), k) for i,k in \
            enumerate(sorted(matrix.keys()))])
    #invert the dict
    int_map = {}
    for i in int_keys:
        int_map[int_keys[i]] = i

    #make a new dict2D object with the integer keys mapped to values instead of
    #the original names
    new_dists = []
    for env1 in matrix:
        for env2 in matrix[env1]:
            new_dists.append((int_map[env1], int_map[env2], matrix[env1][env2]))
    int_map_dists = Dict2D(new_dists)

    #names will be fed into the phylipTable function - it is the int map names
    names = sorted(int_map_dists.keys())
    rows = []
    #populated rows with values based on the order of names
    #the following code will work for a square matrix only
    for index, key1 in enumerate(names):
        row = []
        for key2 in names:
            row.append(str(int_map_dists[key1][key2]))
        rows.append(row)
    input_matrix = phylipMatrix(rows, names)
    #input needs a trailing whitespace or it will fail!
    input_matrix += '\n'

    return input_matrix, int_keys