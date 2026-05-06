def vector_distance(v1, v2):
    """Given 2 vectors of multiple dimensions, calculate the euclidean 
    distance measure between them."""
    dist = 0
    for dim in v1:
        for x in v1[dim]:
            dd = int(v1[dim][x]) - int(v2[dim][x])
            dist = dist + dd**2
    return dist