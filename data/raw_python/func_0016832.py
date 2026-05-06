def redistribute_threads(blockdimx, blockdimy, blockdimz,
    dimx, dimy, dimz):
    """
    Redistribute threads from the Z dimension towards the X dimension.
    Also clamp number of threads to the problem dimension size,
    if necessary
    """

    # Shift threads from the z dimension
    # into the y dimension
    while blockdimz > dimz:
        tmp = blockdimz // 2
        if tmp < dimz:
            break

        blockdimy *= 2
        blockdimz = tmp

    # Shift threads from the y dimension
    # into the x dimension
    while blockdimy > dimy:
        tmp = blockdimy // 2
        if tmp < dimy:
            break

        blockdimx *= 2
        blockdimy = tmp

    # Clamp the block dimensions
    # if necessary
    if dimx < blockdimx:
        blockdimx = dimx

    if dimy < blockdimy:
        blockdimy = dimy

    if dimz < blockdimz:
        blockdimz = dimz

    return blockdimx, blockdimy, blockdimz