def get_next_neighbors(indices, lattice_length, width=0, distance=1,
                       periodic=False):
    """Get the forward neighbors at a given distance of a site or set of sites
    in a lattice.

    :param index: Linear index of operator.
    :type index: int.
    :param lattice_length: The size of the 2D lattice in either dimension
    :type lattice_length: int.
    :param width: Optional parameter to define width.
    :type width: int.
    :param distance: Optional parameter to define distance.
    :type width: int.
    :param periodic: Optional parameter to indicate periodic boundary
                     conditions.
    :type periodic: bool

    :returns: list of int -- the neighbors at given distance in linear index.
    """
    if not isinstance(indices, list):
        indices = [indices]
    if distance == 1:
        return flatten(get_neighbors(index, lattice_length, width, periodic)
                       for index in indices)
    else:
        s1 = set(flatten(get_next_neighbors(get_neighbors(index,
                                                          lattice_length,
                                                          width, periodic),
                                            lattice_length, width, distance-1,
                                            periodic) for index in indices))
        s2 = set(get_next_neighbors(indices, lattice_length, width, distance-1,
                                    periodic))
        return list(s1 - s2)