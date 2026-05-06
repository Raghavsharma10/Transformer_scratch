def get_neighbors(index, lattice_length, width=0, periodic=False):
    """Get the forward neighbors of a site in a lattice.

    :param index: Linear index of operator.
    :type index: int.
    :param lattice_length: The size of the 2D lattice in either dimension
    :type lattice_length: int.
    :param width: Optional parameter to define width.
    :type width: int.
    :param periodic: Optional parameter to indicate periodic boundary
                     conditions.
    :type periodic: bool

    :returns: list of int -- the neighbors in linear index.
    """
    if width == 0:
        width = lattice_length
    neighbors = []
    coords = divmod(index, width)
    if coords[1] < width - 1:
        neighbors.append(index + 1)
    elif periodic and width > 1:
        neighbors.append(index - width + 1)
    if coords[0] < lattice_length - 1:
        neighbors.append(index + width)
    elif periodic:
        neighbors.append(index - (lattice_length - 1) * width)
    return neighbors