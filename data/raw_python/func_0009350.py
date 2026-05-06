def check_contiguity(w, neighbors, leaver):
    """Check if contiguity is maintained if leaver is removed from neighbors


    Parameters
    ----------

    w           : spatial weights object
                  simple contiguity based weights
    neighbors   : list
                  nodes that are to be checked if they form a single \
                          connected component
    leaver      : id
                  a member of neighbors to check for removal


    Returns
    -------

    True        : if removing leaver from neighbors does not break contiguity
                  of remaining set
                  in neighbors
    False       : if removing leaver from neighbors breaks contiguity

    Example
    -------

    Setup imports and a 25x25 spatial weights matrix on a 5x5 square region.

    >>> import libpysal as lps
    >>> w = lps.weights.lat2W(5, 5)

    Test removing various areas from a subset of the region's areas.  In the
    first case the subset is defined as observations 0, 1, 2, 3 and 4. The
    test shows that observations 0, 1, 2 and 3 remain connected even if
    observation 4 is removed.

    >>> check_contiguity(w,[0,1,2,3,4],4)
    True
    >>> check_contiguity(w,[0,1,2,3,4],3)
    False
    >>> check_contiguity(w,[0,1,2,3,4],0)
    True
    >>> check_contiguity(w,[0,1,2,3,4],1)
    False
    >>>
    """

    ids = neighbors[:]
    ids.remove(leaver)
    return is_component(w, ids)